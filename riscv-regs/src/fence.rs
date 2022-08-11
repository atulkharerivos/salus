// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Memory fence instructions. Devices are assumed to be DMA coherent.

#[cfg(all(target_arch = "riscv64", target_os = "none"))]
use core::arch::asm;

// Safety: The `fence` instruction itself does not access memory; it's only side-effect is to
// enforce ordering of surrounding load/store instructions with respect to the `fence`.

/// Orders preceeding memory stores with respect to succeeding memory stores.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
pub fn dma_wmb() {
    unsafe { asm!("fence w,w") };
}

/// Orders preceeding memory loads with respect to succeeding memory loads.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
pub fn dma_rmb() {
    unsafe { asm!("fence r,r") };
}

/// Orders preceeding memory stores with respect to succeeding IO stores.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
pub fn mmio_wmb() {
    // TODO: MMIO writes made from critical sections potentially need a `fence o,w` depending on
    // how the lock is released in order to order the MMIO store with the store to release the
    // lock.
    unsafe { asm!("fence w,o") };
}

/// Orders preceeding IO loads with respect to succeeding memory loads.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
pub fn mmio_rmb() {
    unsafe { asm!("fence i,r") };
}

// Make fence instructions a no-op for testing.
#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
pub fn dma_wmb() {}
#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
pub fn dma_rmb() {}
#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
pub fn mmio_wmb() {}
#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
pub fn mmio_rmb() {}
