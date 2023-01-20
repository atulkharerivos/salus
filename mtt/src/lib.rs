// Copyright (c) 2023 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! This crate provides types and API function related to the Memory Translation Table (MTT).
//! The MTT is maintained by the TSM (Salus), and it used by hardware to distinguish between
//! confidential and non-confidential memory ranges.
#![no_std]

// For testing use the std crate.
#[cfg(test)]
#[macro_use]
extern crate std;

