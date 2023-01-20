// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
use core::marker::PhantomData;
use riscv_pages::{PageAddr, PageSize, Pfn, SupervisorPageAddr, SupervisorPfn, SupervisorPhysAddr};
use spin::{Mutex, Once};

use Error::*;
/// MTT related errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The MTT L2 entry corresponding to the physical address was invalid.
    /// This should be treated as an operational error.
    InvalidL2Entry,
    /// The MTT L1 entry corresponding to the physical address was invalid (encoding of 11b).
    /// This should be treated as an operational error.
    InvalidL1Entry,
    /// The MTT L2 entries for the specified 1GB range failed to pass the invariant that all
    /// 16 consecutive entries (each mapping 64MB) must have the same type.
    ///  This should be treated as an operational error.
    Invalid1GBRange,
    /// The caller specified an operation on a 1GB range, but the L2 entry for the physical
    /// address does not map a 1GB range.
    Non1GBL2Entry,
    /// The caller specified a operation on a 64MB range, but the L2 entry for the physical
    /// address doesn't map a 64MB range.
    Non64MBL2Entry,
    /// The caller specified a operation on a 4K page, but the L2 entry for the physical
    /// address doesn't map a 4K page.
    Non4KL2Entry,
    /// The caller specified an invalid physical address (too large for example)
    InvalidAddress,
    /// The address isn't aligned on a 1G boundary
    InvalidAligment1G,
    /// The address isn't aligned on a 2MB boundary
    InvalidAligment2M,
    /// The range type and length are mismatched
    RangeTypeMismatch,
    /// The specificied page size is yet supported
    UnsupportedPageSize,
    /// The caller specified an invalid number of pages for the range (example: 0,
    /// or would overflow address space)
    InvalidAddressRange,
    /// The platform initialization hasn't been completed. The implementation is expected
    /// to initialize pointers to the MTT L2 and L1 page pool using platform-FW tables.
    PlatformInitNotCompleted,
    /// The platform configuration for the MTT L2 and L1 is invalid
    InvalidPlatformConfiguration,
    /// The platform didn't allocate sufficient L1 page pool pages to map the specified address
    InsufficientL1Pages,
}

/// Holds the result of a MTT operation.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
/// API-related enums for non-confidential and confidential types
pub enum MttMemoryType {
    /// A confidential memory range
    Confidential,
    /// A non-confidential memory range
    NonConfidential,
}

const MTT_L1_ENTRY_COUNT: usize = 512;
/// Struct to represent a L1 page
/// Each 8-byte entry represents 32 4K pages, and the 512 total entries span a total of
/// 64MB of address space.Each page uses a 2-bit encoding, with '00b as non-confidential
/// and '01b as confidential ('11b is illegal).
pub struct MttL1Page {
    base: [u64; MTT_L1_ENTRY_COUNT],
}

struct PlatformMttTables {
    mtt_l2_base: &'static mut [u64],
}

/// Holds references to the platform allocated L2 and L2 tables in a Mutex.
/// This should be constructed by calling init() with the platform allocated backing
/// memory for the L2 and L1 tables.
pub struct PlatformMttInfo {
    inner: Mutex<PlatformMttTables>,
}

// Singleton instance of PlatformMttInfo constructed by PlatformMttInfo::init()
static PLATFORM_MTT_INFO: Once<PlatformMttInfo> = Once::new();

impl PlatformMttInfo {
    /// Creates the singleton PlatformMttInfo instance.
    /// The caller must ensure that:
    /// 1: `mtt_l2_base` is a slice that's backed by platform allocated memory for the MTT L2.
    /// 2: `mtt_l1_page_pool_base` is a slice that's backed by platform allocated memory for
    /// the MTT L1 page pool.
    /// 3: `mtt_l2_base` and `mtt_l1_page_pool_base` must not be aliased elsewhere, and must
    /// remain allocated until a platform reset. The function performs some basic sanity checks
    /// on the passed in parameters.
    pub fn init(mtt_l2_base: &'static mut [u64]) -> Result<()> {
        const MTT_L2_BYTE_SIZE: u64 = 1024 * 1024 * 8;
        // L2 must be 8MB aligned with a length of 8MB, and L1 must be page aligned
        let is_l2_aligned = (mtt_l2_base.as_ptr() as u64) & (MTT_L2_BYTE_SIZE - 1) == 0;
        let is_l2_size_ok = mtt_l2_base.len() as u64 == MTT_L2_BYTE_SIZE / 8;
        if is_l2_aligned && is_l2_size_ok {
            PLATFORM_MTT_INFO.call_once(|| Self {
                inner: Mutex::new(PlatformMttTables { mtt_l2_base }),
            });
            Ok(())
        } else {
            Err(InvalidPlatformConfiguration)
        }
    }

    // Returns a static reference to singleton instance if platform initialization has been completed
    fn get() -> Result<&'static PlatformMttInfo> {
        let platform_mtt_info = PLATFORM_MTT_INFO.get().ok_or(PlatformInitNotCompleted)?;
        Ok(platform_mtt_info)
    }
}

trait MemoryType {}
struct Conf;
struct NonConf;
impl MemoryType for Conf {}
impl MemoryType for NonConf {}

// Enumeration of defined L2 entry types.
// The MTT (Memory Tracking Table) L2 is a 8MB physically contiguous range of memory allocated
// by trusted platform-FW. When enabled in hardware, it partitions memory into non-confidential
// and confidential regions. The enumerations below correspond to the values that can be encoded
// in the 64-bit entries in the table. The entries are encoded as follows:
// |--------------------------------------------------
// |  63:36: Zero | 35:34: Type | 33:0: Info
// |---------------------------------------------------
// The 2-bit encoding for Type is as follows:
// 00b: Non-confidential 1GB range
// 01b: Confidential 1GB range
// 10b: A 64-MB range further described by 16K entries in a 4K page (pointed to by Info[33:0])
// 11b: A 64-MB range composed of 32x2MB sub-ranges, and further described by Info[31:0]
enum MttL2Entry<'a> {
    // The entire 1GB range containing the address is non-confidential.
    // 1GB ranges are mapped using 64MB subranges, and by convention, the invariant is
    // that each of the 16 consecutive entries in the L2 table have the same type.
    NonConfidential1G(Mtt1GEntry<'a, NonConf>),
    // The entire 1GB range containing the address is confidential.
    // 1GB ranges are mapped using 64MB subranges, and by convention, the invariant is
    // that each of the 16 consecutive entries in the L2 table have the same type.
    Confidential1G(Mtt1GEntry<'a, Conf>),
    // The 64-MB range has been partitioned into 2MB regions, and each sub-region can be
    // confidential or non-confidential
    Mixed2M(Mtt64MEntry<'a>),
    // The 64-MB range has been partitioned into 16K regions of 4KB size.
    // Each sub-region can be confidential or non-confidential
    L1NewTable(MttL1Table<'a>),
}
use MttL2Entry::*;

struct MttL2Directory;

impl MttL2Directory {
    const MTT_L2_TYPE_SHIFT: u64 = 34;
    // Bits 33:0
    const L2_INFO_MASK: u64 = 0x0003_ffff_ffff;

    // Returns the index of the MTT L2 entry for the physical address
    fn get_mtt_index(phys_addr: SupervisorPageAddr) -> usize {
        let addr = phys_addr.bits();
        // Bits 45:26
        ((addr & 0x3fff_fc00_0000) >> 26) as usize
    }

    fn entry_for_addr(
        phys_addr: SupervisorPageAddr,
        mtt_l2_base: &mut [u64],
    ) -> Result<MttL2Entry> {
        // Bits 63:36 must be zero
        const L2_ZERO_MASK: u64 = 0xffff_fff0_0000_0000;
        // Bits 35:34
        const L2_TYPE_MASK: u64 = 0x000c_0000_0000;

        let mtt_entry_index = Self::get_mtt_index(phys_addr);
        let value = mtt_l2_base[mtt_entry_index];
        if (value & L2_ZERO_MASK) != 0 {
            return Err(InvalidL2Entry);
        }
        let mtt_l2_type = (value & L2_TYPE_MASK) >> MttL2Directory::MTT_L2_TYPE_SHIFT;

        match mtt_l2_type {
            // INFO must be 0 for 1GB entries
            0 if (value & MttL2Directory::L2_INFO_MASK) == 0 => Ok(NonConfidential1G(
                Mtt1GEntry::new_nc(&mut mtt_l2_base[mtt_entry_index]),
            )),
            1 if (value & MttL2Directory::L2_INFO_MASK) == 0 => Ok(Confidential1G(
                Mtt1GEntry::new_conf(&mut mtt_l2_base[mtt_entry_index]),
            )),
            2 => {
                let pfn = Pfn::supervisor(value & MttL2Directory::L2_INFO_MASK);
                let l1_page_addr = (pfn.bits() >> 12) as *mut MttL1Page;
                // Safety: l1_table is an uniquely owned page that's guaranteed by contract by
                // be not aliased elsewhere by contract.
                // The page will remain allocated until a system system.
                let l1_page = unsafe { &mut *core::ptr::slice_from_raw_parts_mut(l1_page_addr, 1) };

                Ok(L1NewTable(MttL1Table::new(
                    &mut mtt_l2_base[mtt_entry_index],
                    l1_page,
                    pfn,
                )))
            }
            3 => Ok(Mixed2M(Mtt64MEntry::new_mixed_64m(
                &mut mtt_l2_base[mtt_entry_index],
                value,
            ))),
            _ => Err(InvalidL2Entry),
        }
    }

    // Converts 16 consecutive entries for the 1GB range spanning the physical address
    // from MTT type 1GB to MTT type 64-MB. Each of newly created entries will still span
    // the same address range (64MB), but they no longer need to adhere to the conventional
    // invariant that all 16 entries must have the same type.
    fn convert_1g_to_64m(addr: Aligned1GAddr, mtt_l2_base: &mut [u64]) -> Result<()> {
        let mtt_index = Self::get_mtt_index(addr.0);
        match Self::entry_for_addr(addr.0, mtt_l2_base)? {
            Confidential1G(_) => {
                mtt_l2_base[mtt_index..]
                    .iter_mut()
                    .take(16)
                    .for_each(|entry| {
                        Mtt64MEntry::new_conf_64m(entry);
                    });
            }
            NonConfidential1G(_) => {
                mtt_l2_base[mtt_index..]
                    .iter_mut()
                    .take(16)
                    .for_each(|entry| {
                        Mtt64MEntry::new_nc_64m(entry);
                    });
            }
            _ => return Err(Non1GBL2Entry),
        }

        Ok(())
    }
}

struct Mtt1GEntry<'a, T>
where
    T: MemoryType,
{
    entry: &'a mut u64,
    conf_state: PhantomData<T>,
}

impl<'a> Mtt1GEntry<'a, Conf> {
    const MTT_L2_1G_C_TYPE: u64 = 1 << MttL2Directory::MTT_L2_TYPE_SHIFT;
    fn convert_to_nc(self) -> Mtt1GEntry<'a, NonConf> {
        Mtt1GEntry::new_nc(self.entry)
    }

    fn new_conf(entry: &'a mut u64) -> Self {
        *entry = Mtt1GEntry::MTT_L2_1G_C_TYPE;
        Self {
            entry,
            conf_state: PhantomData,
        }
    }
}

impl<'a> Mtt1GEntry<'a, NonConf> {
    const MTT_L2_1G_NC_TYPE: u64 = 0;
    fn convert_to_conf(self) -> Mtt1GEntry<'a, Conf> {
        Mtt1GEntry::new_conf(self.entry)
    }

    fn new_nc(entry: &'a mut u64) -> Self {
        *entry = Mtt1GEntry::MTT_L2_1G_NC_TYPE;
        Self {
            entry,
            conf_state: PhantomData,
        }
    }
}

struct Mtt64MEntry<'a> {
    bit_mask_2m: u32,
    entry: &'a mut u64,
}

impl<'a> Mtt64MEntry<'a> {
    const L2_64M_TYPE: u64 = 3u64 << MttL2Directory::MTT_L2_TYPE_SHIFT;
    // Converts an existing MTT L2 entry representing a 64MB range into a MttL1Entry.
    // The new entry will span the same 64MB range, but it will be broken up into
    // 32-regions of 2MB each. Each of the 2MB regions is mapped to 16 8-bytes entries
    // with 2-bits per 4K page in the region.
    // Note that it's OK to update the L1 page without any concern about a potential
    // race condition with an hardware initiated walk of the MTT. This is because we
    // bare converting from MTT2BPages to MTTL1Dir, and the type of MTT L2 entry remains
    // unchanged until it's atomically updated below following the L1 page write.
    fn split_to_l1_table(self, l1_table: SupervisorPageAddr) -> MttL1Table<'a> {
        let l1_page_addr = l1_table.bits() as *mut MttL1Page;
        let pfn = Pfn::supervisor(l1_table.pfn().bits());
        // Safety: l1_table is an uniquely owned page that's guaranteed by contract by
        // be not aliased elsewhere by contract.
        // The page will remain allocated until a system system.
        let l1_page = unsafe { &mut *core::ptr::slice_from_raw_parts_mut(l1_page_addr, 1) };
        let value = *self.entry;
        for i in 0usize..=31 {
            // Mark the 2MB sub-range (128x16K) as confidential if the bit in the 32-bit mask is set
            let write_value = if (value & (1 << i)) != 0 {
                // 01010101b representing 4-pages of confidential memory
                0x5555_5555_5555_5555u64
            } else {
                0
            };
            l1_page[0]
                .base
                .iter_mut()
                .skip(i * 16)
                .take(16)
                .for_each(|entry| {
                    *entry = write_value;
                });
        }
        MttL1Table::new(self.entry, l1_page, pfn)
    }

    fn new_conf_64m(entry: &'a mut u64) -> Mtt64MEntry<'a> {
        *entry = Mtt64MEntry::L2_64M_TYPE | 0xffff_ffff;
        Self {
            bit_mask_2m: 0xffff_ffff,
            entry,
        }
    }

    fn new_mixed_64m(entry: &'a mut u64, value: u64) -> Mtt64MEntry<'a> {
        let bit_mask_2m = (value & 0xffff_ffff) as u32;
        *entry = Mtt64MEntry::L2_64M_TYPE | (value & MttL2Directory::L2_INFO_MASK);
        Self { bit_mask_2m, entry }
    }

    fn new_nc_64m(entry: &'a mut u64) -> Mtt64MEntry<'a> {
        *entry = Mtt64MEntry::L2_64M_TYPE;
        Self {
            bit_mask_2m: 0,
            entry,
        }
    }

    fn get_bit_index(addr: Aligned2MAddr) -> u64 {
        let addr_bits = addr.0.bits();
        let round_down_64m = addr_bits & !((1024 * 1024 * 64) - 1);
        (addr_bits - round_down_64m) / PageSize::Size2M as u64
    }

    fn set_conf_2m(&mut self, addr: Aligned2MAddr) {
        let bit_index = Self::get_bit_index(addr);
        *self.entry |= 1 << bit_index;
    }

    fn set_nc_2m(&mut self, addr: Aligned2MAddr) {
        let bit_index = Self::get_bit_index(addr);
        *self.entry &= !(1 << bit_index);
    }

    fn is_conf(&self, addr: Aligned2MAddr) -> bool {
        let bit_index = Self::get_bit_index(addr);
        self.bit_mask_2m & (1 << bit_index) != 0
    }
}

/// Enumeration of the defined MTT L1 entry types.
/// The bit encoding is as follows:
/// 00b: The 4K region is non-confidential
/// 01b: The 4K region is confidential
/// 11b: Invalid encoding
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum MttL1EntryType {
    /// The address is mapped to non-confidential 4K page.
    MttNonConfidential4K = 0,
    /// The address is mapped to a confidential 4K page.
    MttConfidential4K = 1,
}

use MttL1EntryType::*;
impl MttL1EntryType {
    const fn from_raw(value: u64) -> Option<MttL1EntryType> {
        match value {
            0 => Some(MttNonConfidential4K),
            1 => Some(MttConfidential4K),
            _ => None,
        }
    }
}

#[allow(dead_code)]
struct MttL1Table<'a> {
    l1_page: &'a mut [MttL1Page],
    entry: &'a mut u64,
}

impl<'a> MttL1Table<'a> {
    fn new(entry: &'a mut u64, l1_page: &'a mut [MttL1Page], l1_table_pfn: SupervisorPfn) -> Self {
        *entry = 2u64 << MttL2Directory::MTT_L2_TYPE_SHIFT | l1_table_pfn.bits();
        Self { l1_page, entry }
    }

    // The L1 is a 4KB page that's used to manage the memory type (non-confidential
    // or confidential) at a 4K-page granularity. This is done by dividing the 64MB
    // region containing the physical address into 16KB regions of 4K each, and the
    // 2-bit encoding in the L1 page pointed to by the L2 entry determines the type.
    // The 12-bit index into the 4K page is computed using the 14-bits [25:12] of
    // the physical address, and then shifting right by 2.
    fn get_l1_index_and_subindex(phys_addr: SupervisorPageAddr) -> (usize, u64) {
        let addr_bits = phys_addr.bits();
        // Bits 25:17 for the 9-bit index
        let l1_index = (addr_bits & 0x03fe_0000) >> 17;
        // 5-bit sub-index (16:12) with 1-bit left shift for the 2-bits per entry
        let sub_index = (addr_bits & 0x1_f000) >> 11;
        (l1_index as usize, sub_index)
    }

    fn is_conf(&self, addr: SupervisorPageAddr) -> bool {
        let (l1_index, sub_index) = Self::get_l1_index_and_subindex(addr);
        let value = self.l1_page[0].base[l1_index];
        let l1_entry =
            MttL1EntryType::from_raw((value >> sub_index) & 0x3).unwrap_or(MttNonConfidential4K);
        matches!(l1_entry, MttConfidential4K)
    }

    fn set_page_type(&mut self, addr: SupervisorPageAddr, memory_type: MttMemoryType) {
        let (l1_index, sub_index) = Self::get_l1_index_and_subindex(addr);
        let l1_page = &mut self.l1_page[0].base;
        let value = l1_page[l1_index];
        let bit_value = if memory_type == MttMemoryType::NonConfidential {
            0u64
        } else {
            1u64
        };
        // Clear the original 2-bits and OR in the new value
        let value = (value & !(3 << sub_index)) | (bit_value << sub_index);
        l1_page[l1_index] = value;
    }
}

enum MttRegionSize {
    Size1G = 1024 * 1024 * 1024,
    Size2M = 1024 * 1024 * 2,
    Size4K = 4096,
}

use MttRegionSize::*;

#[derive(Copy, Clone)]
struct Aligned1GAddr(SupervisorPageAddr);

#[derive(Copy, Clone)]
struct Aligned2MAddr(SupervisorPageAddr);

impl Aligned1GAddr {
    fn new(addr: SupervisorPageAddr) -> Result<Self> {
        if !PageSize::Size1G.is_aligned(addr.bits()) {
            return Err(InvalidAligment1G);
        }
        Ok(Self(addr))
    }
}

impl Aligned2MAddr {
    fn new(addr: SupervisorPageAddr) -> Result<Self> {
        if !PageSize::Size2M.is_aligned(addr.bits()) {
            return Err(InvalidAligment2M);
        }
        Ok(Self(addr))
    }
}

struct MttRange<'a> {
    addr: SupervisorPageAddr,
    len: usize,
    mtt_l2_base: &'a mut [u64],
}

impl<'a> MttRange<'a> {
    fn new(addr: SupervisorPageAddr, len: usize, mtt_l2_base: &'a mut [u64]) -> Result<Self> {
        const MAX_PAGE_ADDR: u64 = 0x3fff_ffff_f000;
        if (len % 4096) != 0 {
            return Err(InvalidAddressRange);
        }

        let range_end = addr
            .checked_add_pages(len as u64 / 4096)
            .ok_or(InvalidAddressRange)?;
        if range_end.bits() & !MAX_PAGE_ADDR != 0 {
            return Err(InvalidAddressRange);
        }

        Ok(Self {
            addr,
            len,
            mtt_l2_base,
        })
    }

    fn set_1g_region_type(
        &mut self,
        addr: SupervisorPageAddr,
        memory_type: MttMemoryType,
    ) -> Result<bool> {
        let l2_entry = MttL2Directory::entry_for_addr(addr, self.mtt_l2_base)?;
        let mtt_updated = match l2_entry {
            Confidential1G(c_1g_entry) => {
                if memory_type == MttMemoryType::NonConfidential {
                    c_1g_entry.convert_to_nc();
                    true
                } else {
                    false
                }
            }
            NonConfidential1G(nc_1g_entry) => {
                if memory_type == MttMemoryType::Confidential {
                    nc_1g_entry.convert_to_conf();
                    true
                } else {
                    false
                }
            }
            _ => return Err(RangeTypeMismatch),
        };

        Ok(mtt_updated)
    }

    fn set_2m_region_type(
        &mut self,
        addr: SupervisorPageAddr,
        memory_type: MttMemoryType,
    ) -> Result<bool> {
        let aligned_2m_addr = Aligned2MAddr::new(addr)?;
        let l2_entry = MttL2Directory::entry_for_addr(addr, self.mtt_l2_base)?;
        let is_conf = match l2_entry {
            Confidential1G(_) => true,
            NonConfidential1G(_) => false,
            Mixed2M(mixed_2m_entry) => mixed_2m_entry.is_conf(aligned_2m_addr),
            _ => return Err(RangeTypeMismatch),
        };

        let mtt_needs_update = (is_conf && memory_type == MttMemoryType::NonConfidential)
            || (!is_conf && memory_type == MttMemoryType::Confidential);

        if mtt_needs_update {
            let l2_entry = MttL2Directory::entry_for_addr(addr, self.mtt_l2_base)?;
            match l2_entry {
                Confidential1G(_) | NonConfidential1G(_) => {
                    let aligned_1g = PageAddr::new(SupervisorPhysAddr::supervisor(
                        PageSize::Size1G.round_down(addr.bits()),
                    ))
                    .ok_or(InvalidAddress)?;
                    let aligned_1g_addr = Aligned1GAddr::new(aligned_1g)?;
                    MttL2Directory::convert_1g_to_64m(aligned_1g_addr, self.mtt_l2_base)?;
                    match MttL2Directory::entry_for_addr(aligned_2m_addr.0, self.mtt_l2_base)? {
                        Mixed2M(mut l2_entry) => {
                            if memory_type == MttMemoryType::Confidential {
                                l2_entry.set_conf_2m(aligned_2m_addr);
                            } else {
                                l2_entry.set_nc_2m(aligned_2m_addr);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                Mixed2M(mut mixed_2m_entry) => {
                    if memory_type == MttMemoryType::Confidential {
                        mixed_2m_entry.set_conf_2m(aligned_2m_addr);
                    } else {
                        mixed_2m_entry.set_nc_2m(aligned_2m_addr);
                    }
                }
                L1NewTable(_) => return Err(RangeTypeMismatch),
            }
        }
        Ok(mtt_needs_update)
    }

    fn set_4k_region_type(
        &mut self,
        addr: SupervisorPageAddr,
        memory_type: MttMemoryType,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
    ) -> Result<bool> {
        let l2_entry = MttL2Directory::entry_for_addr(addr, self.mtt_l2_base)?;
        let is_conf = match l2_entry {
            Confidential1G(_) => true,
            NonConfidential1G(_) => false,
            Mixed2M(mixed_2m_entry) => mixed_2m_entry.is_conf(Aligned2MAddr::new(addr)?),
            L1NewTable(l1_table) => l1_table.is_conf(addr),
        };

        let mtt_needs_update = (is_conf && memory_type == MttMemoryType::NonConfidential)
            || (!is_conf && memory_type == MttMemoryType::Confidential);

        if mtt_needs_update {
            let l2_entry = MttL2Directory::entry_for_addr(addr, self.mtt_l2_base)?;
            let l1_page = if !matches!(l2_entry, L1NewTable(_)) {
                get_l1_page()
            } else {
                None
            };

            let mut l1_table = match l2_entry {
                Confidential1G(_) | NonConfidential1G(_) if let Some(l1_page) = l1_page => {
                    let aligned_1g = PageAddr::new(SupervisorPhysAddr::supervisor(
                        PageSize::Size1G.round_down(addr.bits()),
                    ))
                    .ok_or(InvalidAddress)?;
                    let aligned_1g_addr = Aligned1GAddr::new(aligned_1g)?;
                    MttL2Directory::convert_1g_to_64m(aligned_1g_addr, self.mtt_l2_base)?;
                    match MttL2Directory::entry_for_addr(addr, self.mtt_l2_base)? {
                        Mixed2M(mixed_2m_entry) => {
                            mixed_2m_entry.split_to_l1_table(l1_page)
                        },
                        _ => unreachable!()
                    }
                },
                Mixed2M(mixed_2m_entry) if let Some(l1_page) = l1_page => {
                    mixed_2m_entry.split_to_l1_table(l1_page)
                },
                L1NewTable(l1_table) => {l1_table},
                _ => return Err(InsufficientL1Pages)
            };

            l1_table.set_page_type(addr, memory_type);
        }
        Ok(mtt_needs_update)
    }

    // Updates the memory type for the physical address in the MTT.
    // The MTT update is skipped if the page is already contained in a spanning
    // range of the same type.
    // The desired page size is assumed to be 4K, and larger spans are broken
    // down into smaller regions.
    fn set_page_range_memory_type(
        &mut self,
        addr: SupervisorPageAddr,
        region_size: MttRegionSize,
        memory_type: MttMemoryType,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
    ) -> Result<bool> {
        let mtt_updated = match region_size {
            Size1G => self.set_1g_region_type(addr, memory_type)?,
            Size2M => self.set_2m_region_type(addr, memory_type)?,
            Size4K => self.set_4k_region_type(addr, memory_type, get_l1_page)?,
        };

        Ok(mtt_updated)
    }
}

impl<'a> Iterator for MttRange<'a> {
    type Item = (SupervisorPageAddr, MttRegionSize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }

        // Unwrap OK: Range has been validated already
        let l2_entry = MttL2Directory::entry_for_addr(self.addr, self.mtt_l2_base).unwrap();

        if matches!(l2_entry, Confidential1G(_) | NonConfidential1G(_))
            && PageSize::Size1G.is_aligned(self.addr.bits())
            && self.len >= PageSize::Size1G as usize
        {
            let addr = self.addr;
            // Unwrap OK: Range has already been validated.
            self.addr = self
                .addr
                .checked_add_pages_with_size(1, PageSize::Size1G)
                .unwrap();
            self.len -= PageSize::Size1G as usize;
            return Some((addr, Size1G));
        }

        // TODO: Allow 4K-entries to map large types
        // This requires a mechanism to free previously allocated L1 tables
        if !matches!(l2_entry, L1NewTable(_))
            && PageSize::Size2M.is_aligned(self.addr.bits())
            && self.len >= PageSize::Size2M as usize
        {
            let addr = self.addr;
            // Unwrap OK: Range has already been validated.
            self.addr = self
                .addr
                .checked_add_pages_with_size(1, PageSize::Size2M)
                .unwrap();
            self.len -= PageSize::Size2M as usize;
            return Some((addr, Size2M));
        }

        let addr = self.addr;
        // Unwrap OK: Range has already been validated.
        self.addr = self.addr.checked_add_pages(1).unwrap();
        self.len -= PageSize::Size4k as usize;
        Some((addr, Size4K))
    }
}

struct Mtt;

impl Mtt {
    fn set_range_type(
        addr: SupervisorPageAddr,
        len: usize,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
        memory_type: MttMemoryType,
    ) -> Result<()> {
        let platform_mtt_info = PlatformMttInfo::get()?;
        let mut inner = platform_mtt_info.inner.lock();
        let mut mtt_range = MttRange::new(addr, len, inner.mtt_l2_base)?;
        while let Some((addr, region_size)) = mtt_range.next() {
            mtt_range.set_page_range_memory_type(addr, region_size, memory_type, get_l1_page)?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn set_confidential(
        addr: SupervisorPageAddr,
        len: usize,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
    ) -> Result<()> {
        Mtt::set_range_type(addr, len, get_l1_page, MttMemoryType::Confidential)
    }

    #[allow(dead_code)]
    pub fn set_nc(
        addr: SupervisorPageAddr,
        len: usize,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
    ) -> Result<()> {
        Mtt::set_range_type(addr, len, get_l1_page, MttMemoryType::NonConfidential)
    }
}
