// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use riscv_pages::{
    PageAddr, PageSize, Pfn, RawAddr, SupervisorPageAddr, SupervisorPfn, SupervisorPhysAddr,
};
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
    /// L1 pages must be confidential
    NonConfidentialL1Page,
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

#[cfg(not(test))]
mod mtt_const {
    pub const L2_TYPE_SHIFT: u64 = 34;
    // Bits 63:36 must be zero
    pub const L2_ZERO_MASK: u64 = 0xffff_fff0_0000_0000;
    // Bits 35:34
    pub const L2_TYPE_MASK: u64 = 0x000c_0000_0000;
    // Bits 33:0
    pub const L2_INFO_MASK: u64 = 0x0003_ffff_ffff;
    // Bits 35:34 for a 1G non-confidential region
    // By convention, this is a total of 16 identical consecutive entries, each mapping 64MB
    pub const L2_1G_NC_TYPE: u64 = 0;
    // Bits 35:34 for a 1G confidential region
    // By convention, this is a total of 16 identical consecutive entries, each mapping 64MB
    pub const L2_1G_C_TYPE: u64 = 1;
    // Bits 35:34 for a 64-MB region mapped using a bit-mask in a 4K L1-page
    pub const L2_L1_TYPE: u64 = 2;
    // Bits 35:34 for a 64-MB region mapped using a 2MB sub-region bitmask
    pub const L2_64M_TYPE: u64 = 3;
    // L2 has 1MB entries
    pub const MTT_L2_ENTRY_COUNT: usize = 1024 * 1024;
    // L2 is 8MB long
    pub const L2_BYTE_SIZE: usize = 1024 * 1024 * 8;
    // A L1 page consists of 512 8-byte entries
    pub const L1_ENTRY_COUNT: usize = 512;
    // Bits 25:17 for the 9-bit index
    pub const L1_INDEX_MASK: u64 = 0x03fe_0000;
    pub const L1_INDEX_SHIFT: u64 = 17;
    // Bits 16:12 for the 5-bit sub-index
    pub const L1_SUBINDEX_MASK: u64 = 0x1_f000;
    // 1-bit left shift for the 2-bits per entry for bits (16:12)
    pub const L1_SUBINDEX_SHIFT: u64 = 11;
    // 2-bits per entry
    pub const L1_ENTRY_MASK: u64 = 0x3;
}

// This is to help facilitate testing. The issue is that when testing, we use the PFN
// of the VA for the L1-page, and since it tends to be very large (unlike the actual PA),
// it might sets bits 35:34 when converting to a L1 page (thereby converting the L2 entry
// into a different type). We avoid the issue by shifting everything left by 4.
#[cfg(test)]
mod mtt_const {
    pub const L2_TYPE_SHIFT: u64 = 38;
    // Bits 63:40 must be zero
    pub const L2_ZERO_MASK: u64 = 0xffff_ff00_0000_0000;
    // Bits 39:38
    pub const L2_TYPE_MASK: u64 = 0x00c0_0000_0000;
    // Bits 37:0
    pub const L2_INFO_MASK: u64 = 0x003f_ffff_ffff;
    // Bits 39:38 for a 1G non-confidential region
    // By convention, this is a total of 16 identical consecutive entries, each mapping 64MB
    pub const L2_1G_NC_TYPE: u64 = 0;
    // Bits 39:38 for a 1G confidential region
    // By convention, this is a total of 16 identical consecutive entries, each mapping 64MB
    pub const L2_1G_C_TYPE: u64 = 1;
    // Bits 39:38 for a 64-MB region mapped using a bit-mask in a 4K L1-page
    pub const L2_L1_TYPE: u64 = 2;
    // Bits 39:38 for a 64-MB region mapped using a 2MB sub-region bitmask
    pub const L2_64M_TYPE: u64 = 3;
    // L2 has 1MB entries
    pub const MTT_L2_ENTRY_COUNT: usize = 1024 * 1024;
    // L2 is 8MB long
    pub const L2_BYTE_SIZE: usize = 1024 * 1024 * 8;
    // A L1 page consists of 512 8-byte entries
    pub const L1_ENTRY_COUNT: usize = 512;
    // Bits 25:17 for the 9-bit index
    pub const L1_INDEX_MASK: u64 = 0x03fe_0000;
    pub const L1_INDEX_SHIFT: u64 = 17;
    // Bits 16:12 for the 5-bit sub-index
    pub const L1_SUBINDEX_MASK: u64 = 0x1_f000;
    // 1-bit left shift for the 2-bits per entry for bits (16:12)
    pub const L1_SUBINDEX_SHIFT: u64 = 11;
    // 2-bits per entry
    pub const L1_ENTRY_MASK: u64 = 0x3;
}

use mtt_const::*;

// Struct to represent a L1 page
// Each 8-byte entry represents 32 4K pages, and the 512 total entries span a total of
// 64MB of address space.Each page uses a 2-bit encoding, with '00b as non-confidential
// and '01b as confidential ('11b is illegal).
struct MttL1Page<'a> {
    base: [u64; L1_ENTRY_COUNT],
    marker: PhantomData<&'a u64>,
}

impl<'a> MttL1Page<'a> {
    // Returns a u64 slice that allows access L1 page entries represented by pfn.
    // #Safety:
    // The caller must guarantee that the pfn points to a valid L1 page that isn't
    // aliased elsewhere, and points to confidential memory. The page must remain
    // allocated for the lifetime of the entry in the L2 table.
    unsafe fn from_pfn(pfn: SupervisorPfn) -> Self {
        let l1_page_addr = (pfn.bits() << 12) as *mut MttL1Page;
        let l1_page = &*core::ptr::slice_from_raw_parts_mut(l1_page_addr, 1);
        l1_page[0]
    }
}

/// Holds references to the platform allocated L2 and L2 tables in a Mutex.
/// This should be constructed by calling init() with the platform allocated backing
/// memory for the L2 and L1 tables.
pub struct Mtt {
    inner: Mutex<MttL2Directory>,
}

// Singleton instance of Mtt constructed by Mtt::init()
static PLATFORM_MTT_TABLES: Once<Mtt> = Once::new();

/// Exposes functions to change and query the type of memory ranges in the MTT.
/// Successful invocation requires a preceding call to Mtt::init().
impl Mtt {
    // Performs sanity checks on the platform L2 table
    #[cfg(not(test))]
    fn validate_mtt(&self) -> Result<()> {
        let l2_dir = self.inner.lock();
        let l2_base_addr = l2_dir.mtt_l2_base.as_ptr() as usize;
        let addr = SupervisorPageAddr::new(RawAddr::supervisor(l2_base_addr as u64))
            .ok_or(InvalidAddress)?;
        // TODO: Add additional checks to validate individual L2 entries
        let mut mtt_range = MttRange::new(addr, L2_BYTE_SIZE, l2_dir)?;
        while let Some((addr, region_size, l2_entry)) = mtt_range.next() {
            let region_memory_type = mtt_range.get_region_memory_type(addr, region_size)?;
            if region_memory_type != MttMemoryType::Confidential {
                return Err(InvalidPlatformConfiguration);
            }
        }
        Ok(())
    }

    // Test only NOP validation
    #[cfg(test)]
    fn validate_mtt(&self) -> Result<()> {
        let l2_dir = self.inner.lock();
        // Intended only to silence clippy warning about unused imports while testing
        let l2_base_addr = l2_dir.mtt_l2_base.as_ptr() as u64;
        SupervisorPageAddr::new(RawAddr::supervisor(l2_base_addr)).ok_or(InvalidAddress)?;
        Ok(())
    }

    /// Creates the singleton Mtt instance.
    /// # Safety
    ///
    /// The caller must ensure that:
    /// `mtt_l2_base` is a platform allocated 8MB memory region that isn't
    /// aliased elsewhere and must remain allocated until the platform resets.
    /// The function performs some basic sanity checks on the passed-in parameters.
    pub unsafe fn init(l2_base_addr: u64, len: usize) -> Result<()> {
        if len != L2_BYTE_SIZE {
            return Err(InvalidPlatformConfiguration);
        }
        // L2 must be 8MB aligned
        if l2_base_addr & (L2_BYTE_SIZE as u64 - 1) != 0 {
            return Err(InvalidPlatformConfiguration);
        }
        let mtt_l2_base: &'static mut [u64] =
            core::slice::from_raw_parts_mut(l2_base_addr as *mut u64, MTT_L2_ENTRY_COUNT);
        PLATFORM_MTT_TABLES.call_once(|| Self {
            inner: Mutex::new(MttL2Directory { mtt_l2_base }),
        });
        Mtt::get()?.validate_mtt()
    }

    // Returns a static reference to singleton instance if previously constructed
    fn get() -> Result<&'static Self> {
        let platform_mtt_info = PLATFORM_MTT_TABLES.get().ok_or(PlatformInitNotCompleted)?;
        Ok(platform_mtt_info)
    }

    fn is_range_of_type(
        addr: SupervisorPageAddr,
        len: usize,
        memory_type: MttMemoryType,
    ) -> Result<bool> {
        let mtt = Mtt::get()?;
        let l2_dir = mtt.inner.lock();
        let mut mtt_range = MttRange::new(addr, len, l2_dir)?;
        while let Some((addr, region_size, l2_entry)) = mtt_range.next() {
            let region_memory_type = MttL2Directory::get_memory_type(addr, l2_entry)?;
            if region_memory_type != memory_type {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn set_range_type(
        addr: SupervisorPageAddr,
        len: usize,
        memory_type: MttMemoryType,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
    ) -> Result<bool> {
        let mtt = Mtt::get()?;
        let l2_dir = mtt.inner.lock();
        let mut mtt_range = MttRange::new(addr, len, l2_dir)?;
        let mut mtt_needs_invalidation = false;
        while let Some((addr, region_size, l2_entry)) = mtt_range.next() {
            if mtt_range.set_region_memory_type(addr, region_size, memory_type, get_l1_page)? {
                mtt_needs_invalidation = true;
            }
        }
        Ok(mtt_needs_invalidation)
    }

    /// Marks the region of length `len` starting with `addr` as confidential in
    /// the Memory Tracking Table (MTT). On success, the function returns whether
    /// the caller should make an ECALL to invalidate the MTT.
    /// `len` must be a multiple of a 4K-page size. The region cannot exceed the
    /// maximum physical page address (currently defined as 46-bits of PA).
    /// # Safety
    /// `get_l1_page` will be called to allocate a 4K-page if the function determines
    /// that the region mapping requires a MTT mapping at a 4K granularity.
    /// The callback must ensure that the allocated page isn't aliased and remains allocated
    /// until it's explictly released (TBD), and it must not perform any MTT related
    /// operations as it will result in a deadlock.
    pub unsafe fn set_confidential(
        addr: SupervisorPageAddr,
        len: usize,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
    ) -> Result<bool> {
        Mtt::set_range_type(addr, len, MttMemoryType::Confidential, get_l1_page)
    }

    /// Marks the region of length `len` starting with `addr` as non-confidential in
    /// the Memory Tracking Table (MTT). On success, the function returns whether
    /// the caller should make an ECALL to invalidate the MTT.
    /// `len` must be a multiple of a 4K-page size. The region cannot exceed the
    /// maximum physical page address (currently defined as 46-bits of PA).
    /// # Safety
    /// `get_l1_page` will be called to allocate a 4K-page if the function determines
    /// that the region mapping requires a MTT mapping at a 4K granularity.
    /// The callback must ensure that the allocated page isn't aliased and remains allocated
    /// until it's explictly released (TBD), and it must not perform any MTT related
    /// operations as it will result in a deadlock.
    pub unsafe fn set_non_confidential(
        addr: SupervisorPageAddr,
        len: usize,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
    ) -> Result<bool> {
        Mtt::set_range_type(addr, len, MttMemoryType::NonConfidential, get_l1_page)
    }

    /// Returns where region of length `len` starting with `addr` is confidential in
    /// the Memory Tracking Table (MTT).
    /// 1: `len` must be a multiple of a 4K-page size. The region cannot exceed the
    /// maximum physical page address (currently defined as 46-bits of PA).
    pub fn is_confidential(addr: SupervisorPageAddr, len: usize) -> Result<bool> {
        Self::is_range_of_type(addr, len, MttMemoryType::Confidential)
    }

    /// Returns where region of length `len` starting with `addr` is non-confidential in
    /// the Memory Tracking Table (MTT).
    /// 1: `len` must be a multiple of a 4K-page size. The region cannot exceed the
    /// maximum physical page address (currently defined as 46-bits of PA).
    pub fn is_non_confidential(addr: SupervisorPageAddr, len: usize) -> Result<bool> {
        Self::is_range_of_type(addr, len, MttMemoryType::NonConfidential)
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
    Mixed2M(Mtt64mEntry<'a>),
    // The 64-MB range has been partitioned into 16K regions of 4KB size.
    // Each sub-region can be confidential or non-confidential
    L1Entry(MttL1Entry<'a>),
}

struct MttL2Directory {
    mtt_l2_base: &'static mut [u64],
}

impl MttL2Directory {
    fn get_memory_type(addr: SupervisorPageAddr, l2_entry: MttL2Entry) -> Result<MttMemoryType> {
        fn is_4k_page_conf(l1_entry: MttL1Entry, addr: SupervisorPageAddr) -> bool {
            // Safety: We are deferencing an existing L1 page entry in the MTT L2 table.
            // This entry was created by Salus, or was created by trusted platform-FW,
            // and the pfn corresponding to the page is guaranteed not be aliased and
            // will remain valid for the lifetime of the L2 entry.
            let l1_slice = unsafe { MttL1Page::l1_slice_from_pfn(l1_entry.pfn()) };
            l1_entry.is_conf(addr, l1_slice)
        }

        use MttL2Entry::*;
        match l2_entry {
            Confidential1G(_) => Ok(MttMemoryType::Confidential),
            NonConfidential1G(_) => Ok(MttMemoryType::NonConfidential),
            Mixed2M(mixed_2m_entry) => {
                if mixed_2m_entry.is_conf(Aligned2MAddr::new(addr)?) {
                    Ok(MttMemoryType::Confidential)
                } else {
                    Ok(MttMemoryType::NonConfidential)
                }
            }
            L1Entry(l1_page) => {
                if is_4k_page_conf(l1_entry, addr) {
                    Ok(MttMemoryType::Confidential)
                } else {
                    Ok(MttMemoryType::NonConfidential)
                }
            }
        }
    }

    // Returns the index of the MTT L2 entry for the physical address
    fn get_mtt_index(phys_addr: SupervisorPageAddr) -> usize {
        let addr = phys_addr.bits();
        // Bits 45:26
        ((addr & 0x3fff_fc00_0000) >> 26) as usize
    }

    fn entry_for_addr(&mut self, phys_addr: SupervisorPageAddr) -> Result<MttL2Entry> {
        use MttL2Entry::*;
        let mtt_index = Self::get_mtt_index(phys_addr);
        let value = self.mtt_l2_base[mtt_index];
        if (value & L2_ZERO_MASK) != 0 {
            return Err(InvalidL2Entry);
        }
        let mtt_l2_type = (value & L2_TYPE_MASK) >> L2_TYPE_SHIFT;

        match mtt_l2_type {
            // INFO must be 0 for 1GB entries
            L2_1G_NC_TYPE if (value & L2_INFO_MASK) == 0 => {
                Ok(NonConfidential1G(Mtt1GEntry::new_nc(&mut self.mtt_l2_base[mtt_index..=mtt_index+5])))
            }
            L2_1G_C_TYPE if (value & L2_INFO_MASK) == 0 => {
                Ok(Confidential1G(Mtt1GEntry::new_conf(&mut self.mtt_l2_base[mtt_index..=mtt_index+5])))
            }
            L2_L1_TYPE => {
                let pfn = Pfn::supervisor(value & L2_INFO_MASK);
                // Safety: We are deferencing an existing L1 page entry in the MTT L2 table.
                // This entry was created by Salus, or was created by trusted platform-FW,
                // and the pfn corresponding to the page is guaranteed not be aliased and
                // will remain valid for the lifetime of the L2 entry.
                let l1_page = unsafe { MttL1Page::from_pfn(pfn) };
                Ok(L1Entry(l1_page))
            }
            L2_64M_TYPE => Ok(Mixed2M(Mtt64mEntry::new(&mut self.mtt_l2_base[mtt_index..=mtt_index+1]))),
            _ => Err(InvalidL2Entry),
        }
    }

    // Converts 16 consecutive entries for the 1GB range spanning the physical address
    // from MTT type 1GB to MTT type 64-MB. Each of newly created entries will still span
    // the same address range (64MB), but they no longer need to adhere to the conventional
    // invariant that all 16 entries must have the same type.
    fn convert_1g_to_64m(&mut self, addr: Aligned1GAddr) -> Result<()> {
        use MttL2Entry::*;
        let mtt_index = Self::get_mtt_index(addr.0);
        match self.entry_for_addr(addr.0)? {
            Confidential1G(_) => {
                self.mtt_l2_base[mtt_index..]
                    .iter_mut()
                    .take(16)
                    .enumerate()
                    .for_each(|(index, entry)| {
                        *entry = Mtt64mEntry::new_conf_64m(mtt_index + index).raw();
                    });
            }
            NonConfidential1G(_) => {
                self.mtt_l2_base[mtt_index..]
                    .iter_mut()
                    .take(16)
                    .enumerate()
                    .for_each(|(index, entry)| {
                        *entry = Mtt64mEntry::new_nc_64m(mtt_index + index).raw();
                    });
            }
            _ => return Err(Non1GBL2Entry),
        }

        Ok(())
    }

    // Gets the memory type for region of `region_size` starting with `addr`.
    // `addr` must be aligned on the region boundary.
    fn get_region_memory_type(
        &self,
        addr: SupervisorPageAddr,
        region_size: PageSize,
    ) -> Result<MttMemoryType> {
        match region_size {
            PageSize::Size1G => self.get_1g_memory_type(addr),
            PageSize::Size2M => self.get_2m_memory_type(addr),
            PageSize::Size4k => self.get_4k_memory_type(addr),
            _ => unreachable!(),
        }
    }

    // Sets the memory type for region of `region_size` starting with `addr` to memory_type.
    // `addr` must be aligned on the region boundary. On sucess, the function returns true
    // if the MTT was updated, and false if the region was already of the same type.
    fn set_region_memory_type(
        &mut self,
        addr: SupervisorPageAddr,
        region_size: PageSize,
        memory_type: MttMemoryType,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
    ) -> Result<bool> {
        match region_size {
            PageSize::Size1G => self.set_1g_memory_type(addr, memory_type),
            PageSize::Size2M => self.set_2m_memory_type(addr, memory_type),
            PageSize::Size4k => self.set_4k_memory_type(addr, memory_type, get_l1_page),
            PageSize::Size512G => unreachable!(),
        }
    }

    fn get_1g_memory_type(&self, addr: SupervisorPageAddr) -> Result<MttMemoryType> {
        use MttL2Entry::*;
        let l2_entry = self.entry_for_addr(addr)?;
        match l2_entry {
            Confidential1G(_) => Ok(MttMemoryType::Confidential),
            NonConfidential1G(_) => Ok(MttMemoryType::NonConfidential),
            _ => Err(Non1GBL2Entry),
        }
    }

    // Sets 16 consecutive entries for the 1GB range spanning the physical address
    // from MTT type 1GB from C <-> NC. On success, the return value indicates
    // whether the MTT was updated.
    fn set_1g_memory_type(
        &mut self,
        addr: SupervisorPageAddr,
        memory_type: MttMemoryType,
    ) -> Result<bool> {
        use MttL2Entry::*;
        let _ = Aligned1GAddr::new(addr)?;
        let mtt_updated = match self.entry_for_addr(addr)? {
            Confidential1G(conf_1g) => {
                if memory_type == MttMemoryType::NonConfidential {
                    self.mtt_l2_base[conf_1g.index..]
                        .iter_mut()
                        .take(16)
                        .for_each(|entry| *entry = Mtt1GEntry::raw_nc_1g());
                    true
                } else {
                    false
                }
            }
            NonConfidential1G(nc_1g) => {
                if memory_type == MttMemoryType::Confidential {
                    self.mtt_l2_base[nc_1g.index..]
                        .iter_mut()
                        .take(16)
                        .for_each(|entry| *entry = Mtt1GEntry::raw_conf_1g());
                    true
                } else {
                    false
                }
            }
            _ => return Err(Non1GBL2Entry),
        };
        Ok(mtt_updated)
    }

    fn get_2m_memory_type(&self, addr: SupervisorPageAddr) -> Result<MttMemoryType> {
        use MttL2Entry::*;
        let aligned_2m_addr = Aligned2MAddr::new(addr)?;
        let l2_entry = self.entry_for_addr(addr)?;
        match l2_entry {
            Confidential1G(_) => Ok(MttMemoryType::Confidential),
            NonConfidential1G(_) => Ok(MttMemoryType::NonConfidential),
            Mixed2M(mixed_2m_entry) => {
                if mixed_2m_entry.is_conf(aligned_2m_addr) {
                    Ok(MttMemoryType::Confidential)
                } else {
                    Ok(MttMemoryType::NonConfidential)
                }
            }
            _ => Err(RangeTypeMismatch),
        }
    }

    fn set_2m_memory_type(
        &mut self,
        addr: SupervisorPageAddr,
        memory_type: MttMemoryType,
    ) -> Result<bool> {
        use MttL2Entry::*;
        let aligned_2m_addr = Aligned2MAddr::new(addr)?;
        let is_conf = matches!(self.get_2m_memory_type(addr)?, MttMemoryType::Confidential);
        let mtt_needs_update = (is_conf && memory_type == MttMemoryType::NonConfidential)
            || (!is_conf && memory_type == MttMemoryType::Confidential);

        if mtt_needs_update {
            let l2_entry = self.entry_for_addr(addr)?;
            match l2_entry {
                Confidential1G(_) | NonConfidential1G(_) => {
                    let aligned_1g = PageAddr::new(SupervisorPhysAddr::supervisor(
                        PageSize::Size1G.round_down(addr.bits()),
                    ))
                    .ok_or(InvalidAddress)?;
                    let aligned_1g_addr = Aligned1GAddr::new(aligned_1g)?;
                    self.convert_1g_to_64m(aligned_1g_addr)?;
                    match self.entry_for_addr(aligned_2m_addr.0)? {
                        Mixed2M(l2_entry) => {
                            if memory_type == MttMemoryType::Confidential {
                                l2_entry.set_conf_2m(aligned_2m_addr, self.mtt_l2_base);
                            } else {
                                l2_entry.set_nc_2m(aligned_2m_addr, self.mtt_l2_base);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                Mixed2M(mixed_2m_entry) => {
                    if memory_type == MttMemoryType::Confidential {
                        mixed_2m_entry.set_conf_2m(aligned_2m_addr, self.mtt_l2_base);
                    } else {
                        mixed_2m_entry.set_nc_2m(aligned_2m_addr, self.mtt_l2_base);
                    }
                }
                L1Entry(_) => return Err(RangeTypeMismatch),
            }
        }
        Ok(mtt_needs_update)
    }

    fn get_4k_memory_type(&self, addr: SupervisorPageAddr) -> Result<MttMemoryType> {
        fn is_4k_page_conf(l1_entry: MttL1Entry, addr: SupervisorPageAddr) -> bool {
            // Safety: We are deferencing an existing L1 page entry in the MTT L2 table.
            // This entry was created by Salus, or was created by trusted platform-FW,
            // and the pfn corresponding to the page is guaranteed not be aliased and
            // will remain valid for the lifetime of the L2 entry.
            let l1_slice = unsafe { MttL1Page::l1_slice_from_pfn(l1_entry.pfn()) };
            l1_entry.is_conf(addr, l1_slice)
        }

        use MttL2Entry::*;
        let l2_entry = self.entry_for_addr(addr)?;
        match l2_entry {
            Confidential1G(_) => Ok(MttMemoryType::Confidential),
            NonConfidential1G(_) => Ok(MttMemoryType::NonConfidential),
            Mixed2M(mixed_2m_entry) => {
                if mixed_2m_entry.is_conf(Aligned2MAddr::new(addr)?) {
                    Ok(MttMemoryType::Confidential)
                } else {
                    Ok(MttMemoryType::NonConfidential)
                }
            }
            L1Entry(l1_entry) => {
                if is_4k_page_conf(l1_entry, addr) {
                    Ok(MttMemoryType::Confidential)
                } else {
                    Ok(MttMemoryType::NonConfidential)
                }
            }
        }
    }

    fn set_4k_memory_type(
        &mut self,
        addr: SupervisorPageAddr,
        memory_type: MttMemoryType,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
    ) -> Result<bool> {
        fn set_4k_page_memory_type(
            l1_entry: MttL1Entry,
            addr: SupervisorPageAddr,
            memory_type: MttMemoryType,
        ) {
            let (l1_index, sub_index) = MttL1Entry::get_l1_index_and_subindex(addr);
            // Safety: We are deferencing an existing L1 page entry in the MTT L2 table.
            // This entry was created by Salus, or was created by trusted platform-FW,
            // and the pfn corresponding to the page is guaranteed not be aliased and
            // will remain valid for the lifetime of the L2 entry.
            let l1_slice = unsafe { MttL1Page::l1_slice_from_pfn(l1_entry.pfn()) };
            let value = l1_slice[l1_index];
            let bit_value = if memory_type == MttMemoryType::NonConfidential {
                0u64
            } else {
                1u64
            };
            // Clear the original 2-bits and OR in the new value
            let value = (value & !(L1_ENTRY_MASK << sub_index)) | (bit_value << sub_index);
            l1_slice[l1_index] = value;
        }

        use MttL2Entry::*;
        let is_conf = matches!(self.get_4k_memory_type(addr)?, MttMemoryType::Confidential);
        let mtt_needs_update = (is_conf && memory_type == MttMemoryType::NonConfidential)
            || (!is_conf && memory_type == MttMemoryType::Confidential);

        if mtt_needs_update {
            let l2_entry = self.entry_for_addr(addr)?;
            let l1_entry = match l2_entry {
                Confidential1G(_) | NonConfidential1G(_) => {
                    let l1_page = get_l1_page().ok_or(InsufficientL1Pages)?;
                    let aligned_1g = PageAddr::new(SupervisorPhysAddr::supervisor(
                        PageSize::Size1G.round_down(addr.bits()),
                    ))
                    .ok_or(InvalidAddress)?;
                    let aligned_1g_addr = Aligned1GAddr::new(aligned_1g)?;
                    self.convert_1g_to_64m(aligned_1g_addr)?;
                    match self.entry_for_addr(addr)? {
                        Mixed2M(mixed_2m_entry) => {
                            self.convert_to_l1_entry(l1_page, mixed_2m_entry)
                        }
                        _ => unreachable!(),
                    }
                }
                Mixed2M(mixed_2m_entry) => {
                    let l1_page = get_l1_page().ok_or(InsufficientL1Pages)?;
                    self.convert_to_l1_entry(l1_page, mixed_2m_entry)
                }
                L1Entry(l1_entry) => l1_entry,
            };

            set_4k_page_memory_type(l1_entry, addr, memory_type);
        }
        Ok(mtt_needs_update)
    }
}

struct Mtt1GEntry<'a, T>
where
    T: MemoryType,
{
    entries: &'a mut [u64],
    conf_state: PhantomData<T>,
}

impl<'a, T> Mtt1GEntry<'a, T>
where
    T: MemoryType,
{
    fn split(self) {
        let val = if self.is_conf() {
            Mtt64mEntry::raw_conf_1g_64m()
        } else {
            Mtt64mEntry::raw_nc_1g_64m()
        };
        self.entries.iter_mut().for_each(|entry| *entry = val);
    }

    fn is_conf(&self) -> bool {
        let val = self.entries[0];
        (val >> L2_TYPE_SHIFT) == L2_1G_C_TYPE
    }

    fn toggle_conf_type(self) {
        let val = if self.is_conf() {
            Mtt1GEntry::raw_nc_1g()
        } else {
            Mtt1GEntry::raw_conf_1g()
        };
        self.entries.iter_mut().for_each(|entry| *entry = val);
    }
}

impl<'a> Mtt1GEntry<'a, Conf> {
    fn new_conf(entries: &'a mut [u64]) -> Self {
        Self {
            entries,
            conf_state: PhantomData,
        }
    }

    fn raw_conf_1g() -> u64 {
        L2_1G_C_TYPE << L2_TYPE_SHIFT
    }
}

impl<'a> Mtt1GEntry<'a, NonConf> {
    fn new_nc(entries: &'a mut [u64]) -> Self {
        Self {
            entries,
            conf_state: PhantomData,
        }
    }

    fn raw_nc_1g() -> u64 {
        L2_1G_NC_TYPE << L2_TYPE_SHIFT
    }
}

struct Mtt64mEntry<'a> {
    entry: &'a mut [u64],
}

impl<'a> Mtt64mEntry<'a> {
    fn new(entry: &'a mut [u64]) -> Mtt64mEntry {
        Self { entry }
    }

    fn raw_conf_1g_64m() -> u64 {
        L2_64M_TYPE << L2_TYPE_SHIFT | 0xffff_ffff
    }

    fn raw_nc_1g_64m() -> u64 {
        L2_64M_TYPE << L2_TYPE_SHIFT
    }

    fn raw(&self) -> u64 {
        self.entry[0]
    }

    fn get_bit_index(addr: Aligned2MAddr) -> u64 {
        let addr_bits = addr.0.bits();
        let round_down_64m = addr_bits & !((1024 * 1024 * 64) - 1);
        (addr_bits - round_down_64m) / PageSize::Size2M as u64
    }

    fn set_conf_2m(&mut self, addr: Aligned2MAddr) {
        let bit_index = Self::get_bit_index(addr);
        self.entry[0] |= 1 << bit_index;
    }

    fn set_nc_2m(&mut self, addr: Aligned2MAddr) {
        let bit_index = Self::get_bit_index(addr);
        self.entry[0] &= !(1 << bit_index);
    }

    fn is_conf(&self, addr: Aligned2MAddr) -> bool {
        let bit_index = Self::get_bit_index(addr);
        self.entry[0] & (1 << bit_index) != 0
    }

    // Converts an existing MTT L2 entry representing a 64MB range into a MttL1Entry.
    // The new entry will span the same 64MB range, but it will be broken up into
    // 32-regions of 2MB each. Each of the 2MB regions is mapped to 16 8-bytes entries
    // with 2-bits per 4K page in the region.
    // Note that it's OK to update the L1 page without any concern about a potential
    // race condition with an hardware initiated walk of the MTT. This is because we
    // bare converting from MTT2BPages to MTTL1Dir, and the type of MTT L2 entry remains
    // unchanged until it's atomically updated below following the L1 page write.
    fn split_to_l1_entry(
        self,
        l1_page_addr: SupervisorPageAddr)  {
        let pfn = Pfn::supervisor(l1_page_addr.pfn().bits());
        // Safety: l1_page_addr was explictly allocated to map the 64m region, and has been
        // verified to be in confidential memory. The allocator has made guarantees regarding
        // the aliasing and lifetime of the page.
        let l1_slice = unsafe { MttL1Page::from_pfn(pfn) };
        let value = self.entry[0];
        for i in 0usize..=31 {
            // Mark the 2MB sub-range as confidential if the bit in the 32-bit mask is set
            let write_value = if (value & (1 << i)) != 0 {
                // This represents 32 4K-pages of confidential memory in a L1-page
                0x5555_5555_5555_5555u64
            } else {
                0
            };
            l1_slice.base.iter_mut().skip(i * 16).take(16).for_each(|entry| {
                *entry = write_value;
            });
        }
        self.entry[0] = MttL1Entry::raw_l1_entry(pfn);
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
    MttNonConfidential4K,
    /// The address is mapped to a confidential 4K page.
    MttConfidential4K,
}

impl MttL1EntryType {
    const fn from_raw(value: u64) -> Option<MttL1EntryType> {
        use MttL1EntryType::*;
        match value {
            0 => Some(MttNonConfidential4K),
            1 => Some(MttConfidential4K),
            _ => None,
        }
    }
}

#[allow(dead_code)]
struct MttL1Entry<'a> {
    l1_page: MttL1Page<'a>,
}

impl<'a> MttL1Entry<'a> {
    fn new(l1_page: MttL1Page<'a>) -> Self {
        Self { l1_page }
    }

    fn raw_l1_entry(pfn: SupervisorPfn) -> u64 {
        L2_L1_TYPE << L2_TYPE_SHIFT | pfn.bits()
    }

    // The L1 is a 4KB page that's used to manage the memory type (non-confidential
    // or confidential) at a 4K-page granularity. This is done by dividing the 64MB
    // region containing the physical address into 16KB regions of 4K each, and the
    // 2-bit encoding in the L1 page pointed to by the L2 entry determines the type.
    // The 12-bit index into the 4K page is computed using the 14-bits [25:12] of
    // the physical address, and then shifting right by 2.
    fn get_l1_index_and_subindex(phys_addr: SupervisorPageAddr) -> (usize, u64) {
        let addr_bits = phys_addr.bits();
        let l1_index = (addr_bits & L1_INDEX_MASK) >> L1_INDEX_SHIFT;
        let sub_index = (addr_bits & L1_SUBINDEX_MASK) >> L1_SUBINDEX_SHIFT;
        (l1_index as usize, sub_index)
    }

    fn is_conf(&self, addr: SupervisorPageAddr) -> bool {
        use MttL1EntryType::*;
        let (l1_index, sub_index) = Self::get_l1_index_and_subindex(addr);
        let value = self.l1_page.base[l1_index];
        let l1_entry = MttL1EntryType::from_raw((value >> sub_index) & L1_ENTRY_MASK)
            .unwrap_or(MttNonConfidential4K);
        matches!(l1_entry, MttConfidential4K)
    }
}

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

// Helper for iterating over a MTT range.
struct MttRange<'a> {
    addr: SupervisorPageAddr,
    len: usize,
    l2_dir: spin::MutexGuard<'a, MttL2Directory>,
}

impl<'a> MttRange<'a> {
    fn new(
        addr: SupervisorPageAddr,
        len: usize,
        l2_dir: spin::MutexGuard<'a, MttL2Directory>,
    ) -> Result<Self> {
        const MAX_PAGE_ADDR: u64 = 0x3fff_ffff_f000;
        if !PageSize::Size4k.is_aligned(len as u64) {
            return Err(InvalidAddressRange);
        }

        let range_end = addr
            .checked_add_pages(len as u64 / 4096)
            .ok_or(InvalidAddressRange)?;
        if range_end.bits() & !MAX_PAGE_ADDR != 0 {
            return Err(InvalidAddressRange);
        }

        Ok(Self { addr, len, l2_dir })
    }

    // Gets the memory type for the physical address in the MTT.
    fn get_region_memory_type(
        &mut self,
        addr: SupervisorPageAddr,
        region_size: PageSize,
    ) -> Result<MttMemoryType> {
        self.l2_dir.get_region_memory_type(addr, region_size)
    }

    fn set_region_memory_type(
        &mut self,
        addr: SupervisorPageAddr,
        region_size: PageSize,
        memory_type: MttMemoryType,
        get_l1_page: &mut dyn FnMut() -> Option<SupervisorPageAddr>,
    ) -> Result<bool> {
        self.l2_dir
            .set_region_memory_type(addr, region_size, memory_type, get_l1_page)
    }
}

// Iterator implementation for the range contained in MttRange.
// The fragment length is determined by the remaing number of pages, and the
// type of MTT mapping for the currrent page address. Specifically, if a entry
// type is already of MttL1, we only return a fragment of 4K since we don't have
// a mechanism to free allocated L1 pages. Also, 1G fragments are returned only
// if the mapping is of the same type.
impl<'a> Iterator for MttRange<'a> {
    type Item = (SupervisorPageAddr, PageSize, MttL2Entry<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        use MttL2Entry::*;
        if self.len == 0 {
            return None;
        }

        // Unwrap OK: Range has been validated already
        let l2_entry = self.l2_dir.entry_for_addr(self.addr).unwrap();
        let page_size = match l2_entry {
            Confidential1G(_) | NonConfidential1G(_)
                if PageSize::Size1G.is_aligned(self.addr.bits())
                    && self.len >= PageSize::Size1G as usize =>
            {
                PageSize::Size1G
            }
            Confidential1G(_) | NonConfidential1G(_) | Mixed2M(_)
                if PageSize::Size2M.is_aligned(self.addr.bits())
                    && self.len >= PageSize::Size2M as usize =>
            {
                PageSize::Size2M
            }
            _ => PageSize::Size4k,
        };

        let addr = self.addr;
        // Unwrap OK: Range has already been validated.
        self.addr = self.addr.checked_add_pages_with_size(1, page_size).unwrap();
        self.len -= page_size as usize;
        Some((addr, page_size, l2_entry))
    }
}

#[cfg(test)]
mod tests {
    use super::MttL2Entry::*;
    use super::*;

    fn verify_1g_type(addr: SupervisorPageAddr, memory_type: MttMemoryType) {
        // Unwrap OK: Test case is expected to succeed
        let platform_mtt_info = Mtt::get().unwrap();
        let mut l2_dir = platform_mtt_info.inner.lock();
        let entry_type = l2_dir.entry_for_addr(addr).unwrap();
        if memory_type == MttMemoryType::Confidential {
            assert!(matches!(entry_type, Confidential1G(_)));
        } else {
            assert!(matches!(entry_type, NonConfidential1G(_)));
        }
    }

    fn verify_2m_type(addr: SupervisorPageAddr, memory_type: MttMemoryType) {
        // Unwrap OK: Test case is expected to succeed
        let platform_mtt_info = Mtt::get().unwrap();
        let mut l2_dir = platform_mtt_info.inner.lock();
        let entry_type = l2_dir.entry_for_addr(addr).unwrap();
        assert!(matches!(entry_type, Mixed2M(_)));
        // Unwrap OK: Test case is expected to succeed
        let addr_2m = Aligned2MAddr::new(addr).unwrap();
        match entry_type {
            Mixed2M(mixed_2m) => {
                let is_conf = mixed_2m.is_conf(addr_2m);
                if memory_type == MttMemoryType::Confidential {
                    assert!(is_conf);
                } else {
                    assert!(!is_conf);
                }
            }
            _ => unreachable!(),
        }
    }

    fn verify_4k_type(addr: SupervisorPageAddr, memory_type: MttMemoryType) {
        // Unwrap OK: Test case is expected to succeed
        let platform_mtt_info = Mtt::get().unwrap();
        let mut l2_dir = platform_mtt_info.inner.lock();
        let entry_type = l2_dir.entry_for_addr(addr).unwrap();
        assert!(matches!(entry_type, L1Entry(_)));
        // Unwrap OK: Test case is expected to succeed
        match entry_type {
            L1Entry(l1_entry) => {
                // Safety: Testing purposes only
                let l1_slice = unsafe { MttL1Page::l1_slice_from_pfn(l1_entry.pfn()) };
                let is_conf = l1_entry.is_conf(addr, l1_slice);
                if memory_type == MttMemoryType::Confidential {
                    assert!(is_conf);
                } else {
                    assert!(!is_conf);
                }
            }
            _ => unreachable!(),
        }
    }

    fn test_mtt_ranges() {
        use riscv_pages::PageSize;
        const L1_PAGE_POOL_COUNT: usize = 16;
        static mut MTT_BASE: [u64; MTT_L2_ENTRY_COUNT * 2] = [0; MTT_L2_ENTRY_COUNT * 2];
        let mut l1_index = 0 as usize;
        #[repr(align(4096))]
        struct L1PagePool {
            buffer: [u8; 4096 * L1_PAGE_POOL_COUNT],
        }
        static mut L1_PAGE_POOL: L1PagePool = L1PagePool {
            buffer: [0; 4096 * L1_PAGE_POOL_COUNT],
        };

        let mut l1_allocator = || {
            if l1_index < L1_PAGE_POOL_COUNT {
                let addr = unsafe { &L1_PAGE_POOL.buffer[l1_index * 4096] as *const u8 as u64 };
                let l1_page = SupervisorPageAddr::new(RawAddr::supervisor(addr)).unwrap();
                l1_index = 1;
                Some(l1_page)
            } else {
                None
            }
        };

        // Safety: Testing purposes only
        unsafe {
            let l2_base = MTT_BASE.as_ptr() as u64;
            const L2_ALIGNMENT: u64 = 1024 * 1024 * 8;
            let aligned_mtt_base = (l2_base + L2_ALIGNMENT - 1) & !(L2_ALIGNMENT - 1);
            // Unwrap OK: Test case is expected to succeed
            Mtt::init(aligned_mtt_base, L2_BYTE_SIZE).unwrap();
        }

        // Unwrap OK: Alignment is guaranteed by construction
        let addr = PageAddr::new(SupervisorPhysAddr::supervisor(0)).unwrap();

        let len_1g = PageSize::Size1G as usize;
        let len_2m = PageSize::Size2M as usize;
        let len_4k = PageSize::Size4k as usize;

        // Safety: Testing purposes only
        unsafe {
            verify_1g_type(addr, MttMemoryType::NonConfidential);

            // Unwrap OK: Test case is expected to succeed
            Mtt::set_non_confidential(addr, len_2m, &mut l1_allocator).unwrap();
            verify_1g_type(addr, MttMemoryType::NonConfidential);
            // Unwrap OK: Test case is expected to succeed
            Mtt::set_non_confidential(addr, len_4k, &mut l1_allocator).unwrap();
            verify_1g_type(addr, MttMemoryType::NonConfidential);

            Mtt::set_confidential(addr, len_1g, &mut l1_allocator).unwrap();
            for i in 0..=15 {
                let addr_incr = addr
                    .checked_add_pages_with_size(i * 32, PageSize::Size2M)
                    .unwrap();
                verify_1g_type(addr_incr, MttMemoryType::Confidential);
            }

            // Unwrap OK: Test case is expected to succeed
            Mtt::set_non_confidential(addr, len_1g, &mut l1_allocator).unwrap();
            verify_1g_type(addr, MttMemoryType::NonConfidential);
            for i in 0..=15 {
                let addr_incr = addr
                    .checked_add_pages_with_size(i * 32, PageSize::Size2M)
                    .unwrap();
                verify_1g_type(addr_incr, MttMemoryType::NonConfidential);
            }

            Mtt::set_confidential(addr, len_1g, &mut l1_allocator).unwrap();
            Mtt::set_confidential(addr, len_2m, &mut l1_allocator).unwrap();
            verify_1g_type(addr, MttMemoryType::Confidential);
            Mtt::set_confidential(addr, len_4k, &mut l1_allocator).unwrap();
            verify_1g_type(addr, MttMemoryType::Confidential);

            // This will break the 1GB regions into 64MB ranges
            // Unwrap OK: Test case is expected to succeed
            Mtt::set_non_confidential(addr, len_1g, &mut l1_allocator).unwrap();
            verify_1g_type(addr, MttMemoryType::NonConfidential);
            Mtt::set_confidential(addr, len_2m, &mut l1_allocator).unwrap();
            verify_2m_type(addr, MttMemoryType::Confidential);
            let addr_incr = addr
                .checked_add_pages_with_size(1, PageSize::Size2M)
                .unwrap();
            verify_2m_type(addr_incr, MttMemoryType::NonConfidential);
            let addr_incr = addr
                .checked_add_pages_with_size(64, PageSize::Size2M)
                .unwrap();
            verify_2m_type(addr_incr, MttMemoryType::NonConfidential);
            Mtt::set_non_confidential(addr, len_2m, &mut l1_allocator).unwrap();
            verify_2m_type(addr, MttMemoryType::NonConfidential);

            let addr = addr
                .checked_add_pages_with_size(1, PageSize::Size1G)
                .unwrap();
            verify_1g_type(addr, MttMemoryType::NonConfidential);
            // Unwrap OK: Test case is expected to succeed
            Mtt::set_non_confidential(addr, len_4k, &mut l1_allocator).unwrap();
            verify_1g_type(addr, MttMemoryType::NonConfidential);
            Mtt::set_confidential(addr, len_4k, &mut l1_allocator).unwrap();
            verify_4k_type(addr, MttMemoryType::Confidential);
            let addr_incr = addr.checked_add_pages(1).unwrap();
            verify_4k_type(addr_incr, MttMemoryType::NonConfidential);

            let addr = addr
                .checked_add_pages_with_size(1, PageSize::Size1G)
                .unwrap();
            verify_1g_type(addr, MttMemoryType::NonConfidential);
            Mtt::set_confidential(addr, len_2m, &mut l1_allocator).unwrap();
            verify_2m_type(addr, MttMemoryType::Confidential);
            Mtt::set_non_confidential(addr, len_4k * 4, &mut l1_allocator).unwrap();
            verify_4k_type(addr, MttMemoryType::NonConfidential);
            for i in 0..4 {
                let addr_incr = addr.checked_add_pages(i).unwrap();
                verify_4k_type(addr_incr, MttMemoryType::NonConfidential);
            }

            let addr_incr = addr.checked_add_pages(5).unwrap();
            verify_4k_type(addr_incr, MttMemoryType::Confidential);
            assert!(Mtt::is_non_confidential(addr, len_4k * 4 as usize).unwrap());
            assert!(!Mtt::is_non_confidential(addr, len_4k * 5 as usize).unwrap());
            assert!(Mtt::is_confidential(addr_incr, len_2m - (len_4k * 5) as usize).unwrap());
        }
    }

    #[test]
    fn test_mtt() {
        test_mtt_ranges();
    }
}
