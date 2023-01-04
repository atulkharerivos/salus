// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use drivers::{pmu, CpuInfo};
use riscv_regs::{
    defs::shpmevent, LocalRegisterCopy, RiscvCsrInterface, Writeable, CSR, CSR_CYCLE,
};
use s_mode_utils::print::*;
use sbi::{
    Error as SbiError, PmuCounterConfigFlags, PmuCounterStartFlags, PmuCounterStopFlags,
    PmuEventType, Result as SbiResult,
};
use PmuCounterState::*;

#[derive(Default, Copy, Clone)]
struct CounterMaskIter {
    counter_index: u64,
    counter_mask: u64,
    mask_index: u64,
}

impl CounterMaskIter {
    fn new(counter_index: u64, counter_mask: u64) -> Self {
        Self {
            counter_index,
            counter_mask,
            mask_index: 0,
        }
    }
}

// Convenience iterator to return counter_state indexes relative to counter_index
// if the counter_mask bit is set (assumes sanitized input).
impl Iterator for CounterMaskIter {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.counter_mask == 0 {
                break None;
            }
            let result = self.counter_index + self.mask_index;
            let mask_bit_set = self.counter_mask & (1 << self.mask_index);
            self.counter_mask &= !(1 << self.mask_index);
            self.mask_index += 1;
            if mask_bit_set != 0 {
                break Some(result as usize);
            }
        }
    }
}

#[derive(Default, Copy, Clone)]
struct CounterState {
    value: u64,
    config_flags: PmuCounterConfigFlags,
    event_type: PmuEventType,
    event_data: u64,
}

#[derive(Copy, Clone)]
enum PmuCounterState {
    NotConfigured,
    Configured(CounterState),
    Started(CounterState),
    Poisoned(CounterState),
}

impl Default for PmuCounterState {
    fn default() -> Self {
        Self::NotConfigured
    }
}

pub struct VmPmuState {
    // Stores information about the current state of PMU counters.
    counter_state: [PmuCounterState; drivers::pmu::MAX_HARDWARE_COUNTERS],
}

impl Default for VmPmuState {
    fn default() -> Self {
        Self {
            counter_state: [PmuCounterState::default(); drivers::pmu::MAX_HARDWARE_COUNTERS],
        }
    }
}

impl VmPmuState {
    // Sets the bit to enable access to the CSR for counter_index
    fn set_hcounteren_bit(counter_index: u64) {
        // Unwrap ok: Guaranteed to succeed since we have already tested the condition in the call
        // chain
        let pmu_info = pmu::PmuInfo::get().unwrap();
        let csr = pmu_info.counter_index_to_csr(counter_index).unwrap();
        CSR.hcounteren
            .read_and_set_bits(1 << (csr - CSR_CYCLE as u64));
    }

    // Clears the bit to enable access to the CSR for counter_index
    fn clear_hcounteren_bit(counter_index: u64) {
        // Unwrap ok: Guaranteed to succeed since we have already tested the condition in the call
        // chain
        let pmu_info = pmu::PmuInfo::get().unwrap();
        let csr = pmu_info.counter_index_to_csr(counter_index).unwrap();
        CSR.hcounteren
            .read_and_clear_bits(1 << (csr - CSR_CYCLE as u64));
    }

    // Updates internal state for PMU counters.
    // This should be called following a successful SBI call to configure counters.
    fn update_configured_counter(
        &mut self,
        counter_index: u64,
        config_flags: PmuCounterConfigFlags,
        event_type: PmuEventType,
        event_data: u64,
    ) -> SbiResult<()> {
        use PmuCounterState::*;
        let new_state = CounterState {
            config_flags,
            event_type,
            event_data,
            value: 0,
        };
        let pmu_info = pmu::PmuInfo::get()?;
        // Ensure that platform configuration returned a valid counter index.
        pmu_info
            .get_counter_info(counter_index)
            .map_err(|_| SbiError::NotSupported)?;
        let state = &mut self.counter_state[counter_index as usize];
        match state {
            // If skip_match is set, counter must already be configured
            Configured(c) | Started(c) if config_flags.is_skip_match() => {
                if config_flags.is_auto_start() {
                    Self::set_hcounteren_bit(counter_index);
                    *state = Started(*c);
                }
                Ok(())
            }
            NotConfigured if !config_flags.is_skip_match() => {
                *state = if config_flags.is_auto_start() {
                    Self::set_hcounteren_bit(counter_index);
                    Started(new_state)
                } else {
                    Configured(new_state)
                };
                Ok(())
            }
            _ => Err(SbiError::InvalidParam),
        }
    }

    // Updates internal state for PMU counters. Assumes a sanitized counter_index and counter_mask.
    // This should be called following a successful SBI call to start counters.
    fn update_started_counters(&mut self, counter_index: u64, counter_mask: u64) {
        use PmuCounterState::*;
        let bitmask_iter = CounterMaskIter::new(counter_index, counter_mask);
        for i in bitmask_iter {
            if let Configured(c) = self.counter_state[i] {
                self.counter_state[i] = Started(c);
                Self::set_hcounteren_bit(i as u64);
            }
        }
    }

    // Updates internal state for PMU counters. Assumes a sanitized counter_index and counter_mask.
    // This should be called following a successful SBI call to stop counters.
    fn update_stopped_counters(
        &mut self,
        counter_index: u64,
        counter_mask: u64,
        stop_flags: PmuCounterStopFlags,
    ) {
        use PmuCounterState::*;
        let bitmask_iter = CounterMaskIter::new(counter_index, counter_mask);
        for i in bitmask_iter {
            let state = &mut self.counter_state[i];
            let is_started_counter = matches!(state, Started(_));
            match state {
                // Deliberately more permissive since the implementation permits
                // operations even on stopped counters (example: stop_flag_reset).
                Configured(c) | Started(c) => {
                    if stop_flags.is_reset_flag() {
                        Self::clear_hcounteren_bit(i as u64);
                        *state = NotConfigured;
                    } else {
                        if is_started_counter {
                            c.value = VmPmuState::read_counter_csr(i as u64);
                        }
                        *state = Configured(*c);
                    }
                }
                _ => {}
            }
        }
    }

    // Returns a filtered counter_mask if the PMU counter range can be started.
    fn get_startable_counter_range(&self, counter_index: u64, counter_mask: u64) -> SbiResult<u64> {
        use PmuCounterState::*;
        let pmu_info = pmu::PmuInfo::get()?;
        let counter_mask = pmu_info.filter_counter_mask(counter_index, counter_mask)?;
        let mut bitmask_iter = CounterMaskIter::new(counter_index, counter_mask);
        bitmask_iter
            .find(|i| matches!(self.counter_state[*i], Configured(_)))
            .map_or_else(|| Err(SbiError::InvalidParam), |_| Ok(counter_mask))
    }

    // Returns a filtered counter mask if the PMU counter range can be stopped.
    fn get_stoppable_counter_range(&self, counter_index: u64, counter_mask: u64) -> SbiResult<u64> {
        let pmu_info = pmu::PmuInfo::get()?;
        let counter_mask = pmu_info.filter_counter_mask(counter_index, counter_mask)?;
        // The current PMU driver attempts to stop counters before configuration since some platform
        // counters are automatically started. If we check for configured counters at this point,
        // the subsequent call to start counters will fail with an AlreadyStarted error.
        Ok(counter_mask)
    }

    // Returns a filtered counter_mask if the PMU counter range can be configured.
    fn get_configurable_counter_range(
        &self,
        counter_index: u64,
        counter_mask: u64,
        config_flags: PmuCounterConfigFlags,
    ) -> SbiResult<u64> {
        use PmuCounterState::*;
        let pmu_info = pmu::PmuInfo::get()?;
        let counter_mask = pmu_info.filter_counter_mask(counter_index, counter_mask)?;
        if !config_flags.is_skip_match() {
            let mut bitmask_iter = CounterMaskIter::new(counter_index, counter_mask);
            bitmask_iter
                .find(|i| matches!(self.counter_state[*i], NotConfigured))
                .map_or_else(|| Err(SbiError::InvalidParam), |_| Ok(counter_mask))
        } else {
            // If skip_match is set, the counter must already be configured
            let state = self
                .counter_state
                .get(counter_index as usize)
                .ok_or(SbiError::InvalidParam)?;
            if matches!(state, Started(_) | Configured(_)) {
                Ok(counter_mask)
            } else {
                Err(SbiError::InvalidParam)
            }
        }
    }

    fn read_counter_csr(counter_index: u64) -> u64 {
        // Unwrap ok: Guaranteed to succeed since we have already tested the condition in the call
        // chain
        let pmu_info = pmu::PmuInfo::get().unwrap();
        // Unwrap ok: The CSR for a configured counter must be valid
        let csr = pmu_info.counter_index_to_csr(counter_index).unwrap();
        CSR.hpmcounter[(csr - CSR_CYCLE as u64) as usize].get_value()
    }

    /// Returns the cached value for a PMU CSR. We return 0 for counters that couldn't be
    /// configured or started on the resume path.
    pub fn get_cached_csr_value(&self, csr: u64) -> SbiResult<u64> {
        use PmuCounterState::*;
        let pmu_info = pmu::PmuInfo::get()?;
        let counter_index = pmu_info.csr_to_counter_index(csr)?;
        match self.counter_state[counter_index as usize] {
            Configured(c) => Ok(c.value),
            Poisoned(c) => Ok(c.value),
            _ => Err(SbiError::Failed),
        }
    }

    fn reset_all_counters(&mut self, counter_mask: u64) -> SbiResult<()> {
        if counter_mask != 0 {
            self.stop_counters_internal(
                0,
                counter_mask,
                PmuCounterStopFlags::default().set_reset_flag(),
            )
            .or_else(|e| {
                // Treat already stopped error as success
                if matches!(e, SbiError::AlreadyStopped) {
                    Ok(())
                } else {
                    Err(e)
                }
            })
        } else {
            Ok(())
        }
    }

    /// Saves the internal state for PMU counters. Stops started counters, and resets all configured
    /// counters. This should be called in anticipation of an outbound context switch.
    pub fn save_counters(&mut self) {
        use PmuCounterState::*;

        if let Ok(pmu_info) = pmu::PmuInfo::get() {
            let num_counters = pmu_info.get_num_counters() as usize;
            let mut counter_mask = 0;
            for (i, state) in self.counter_state.iter_mut().take(num_counters).enumerate() {
                let include_counter = matches!(state, Configured(_) | Started(_));
                if include_counter {
                    counter_mask |= 1 << i;
                    if let Started(c) = state {
                        c.value = VmPmuState::read_counter_csr(i as u64);
                    }
                }
            }

            let result = self.reset_all_counters(counter_mask);
            if result.is_err() {
                println!(
                    "Warning: PMU failed to reset counters with mask {counter_mask:x}, {result:?}"
                );
            }
        }
    }

    fn resume_counter(&mut self, counter_index: u64, c: &CounterState) -> SbiResult<()> {
        let start_flags = PmuCounterStartFlags::default().set_init_value();
        self.start_counters_internal(counter_index, 0x1, start_flags, c.value)
            .map(|_| VmPmuState::set_hcounteren_bit(counter_index))
    }

    fn configure_counter(&mut self, counter_index: u64, c: &CounterState) -> SbiResult<u64> {
        let config_flags = if !CpuInfo::get().has_sscdeleg() {
            c.config_flags
                .unset_auto_start()
                .unset_skip_match()
                .unset_clear_value()
        } else {
            // Set skip_match if counter delegation is enabled
            c.config_flags
                .unset_auto_start()
                .set_skip_match()
                .unset_clear_value()
        };

        self.configure_matching_counters_internal(
            counter_index,
            0x1,
            config_flags,
            c.event_type,
            c.event_data,
        )
    }

    /// Restores configured PMU counters, restarts started counters and enables CSR access as
    /// necessary. This should be called in anticipation of an inbound context switch.
    pub fn restore_counters(&mut self) {
        use PmuCounterState::*;

        if let Ok(pmu_info) = pmu::PmuInfo::get() {
            let num_counters = pmu_info.get_num_counters() as usize;
            for i in 0..num_counters {
                let state = &self.counter_state[i].clone();
                let counter_index = i as u64;
                let is_started_counter = matches!(state, Started(_));
                match state {
                    Configured(c) | Started(c) => {
                        let result = self.configure_counter(counter_index, c).and_then(|_| {
                            if is_started_counter {
                                self.resume_counter(counter_index, c)
                            } else {
                                Ok(())
                            }
                        });
                        if result.is_err() {
                            self.counter_state[i] = Poisoned(*c);
                            println!(
                                "Warning: Failed to restore counter {counter_index}, {result:?}"
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn configure_delegated_matching_counters(
        &mut self,
        counter_index: u64,
        counter_mask: u64,
        config_flags: PmuCounterConfigFlags,
        event_type: PmuEventType,
        event_data: u64,
    ) -> SbiResult<u64> {
        use sbi::PmuHardware::*;
        fn configure_counter_csrs(
            config_flags: PmuCounterConfigFlags,
            event_type: PmuEventType,
            event_data: u64,
            counter_index: u64,
        ) {
            let mut shpmevent = LocalRegisterCopy::<u64, shpmevent::Register>::new(0);
            // Translate uinh/sinh
            if config_flags.is_uinh() {
                shpmevent.modify(shpmevent::vuinh.val(1));
            }
            if config_flags.is_sinh() {
                shpmevent.modify(shpmevent::vsinh.val(1));
            }
            shpmevent.modify(shpmevent::sinh.val(1));
            // TODO: The mapping of events to platform counters is still under discussion.
            // This will need to be revisited after the standard has been finalized.
            match event_type {
                PmuEventType::Hardware(_) | PmuEventType::Cache(_) => {
                    shpmevent.modify(shpmevent::event_selector.val(event_type.raw()))
                }
                PmuEventType::RawEvent => {
                    shpmevent.modify(shpmevent::event_selector.val(event_data))
                }
                _ => unreachable!(),
            }
            CSR.scounterselect.set(counter_index);
            if !config_flags.is_skip_match() {
                CSR.shpmevent.set(shpmevent.get());
            }
            if config_flags.is_clear_value() {
                CSR.shpmcounter.set(0);
            }
            if config_flags.is_auto_start() {
                CSR.scountinhibit.read_and_clear_bits(1 << counter_index);
            }
        }

        let pmu_info = pmu::PmuInfo::get()?;
        let counter_mask = pmu_info.filter_counter_mask(counter_index, counter_mask)?;
        let index = if !config_flags.is_skip_match() {
            let fixed_index = match event_type {
                PmuEventType::Hardware(CpuCycles) => Some(0),
                PmuEventType::Hardware(Instructions) => Some(2),
                PmuEventType::Firmware(_) => return Err(SbiError::InvalidParam),
                _ => None,
            };
            let iter = CounterMaskIter::new(counter_index, counter_mask);
            let found_index = iter
                .filter(|i| {
                    (fixed_index.is_none() && *i > 2)
                        || (fixed_index == Some(*i as u64)
                            && pmu_info.get_counter_info(*i as u64).is_ok())
                })
                .find(|i| matches!(self.counter_state[*i], PmuCounterState::NotConfigured))
                .ok_or(SbiError::InvalidParam)?;
            Ok(found_index as u64)
        } else {
            // If skip_match is set, the counter must already be configured
            self.counter_state
                .get(counter_index as usize)
                .filter(|s| matches!(s, Started(_) | Configured(_)))
                .ok_or(SbiError::InvalidParam)?;
            Ok(counter_index)
        }?;

        configure_counter_csrs(config_flags, event_type, event_data, index);
        Ok(index)
    }

    fn configure_matching_counters_internal(
        &mut self,
        counter_index: u64,
        counter_mask: u64,
        config_flags: PmuCounterConfigFlags,
        event_type: PmuEventType,
        event_data: u64,
    ) -> SbiResult<u64> {
        // Translate uinh/sinh from VM
        let mut config_flags = config_flags;
        if config_flags.is_uinh() {
            config_flags = config_flags.set_vuinh();
        }
        if config_flags.is_sinh() {
            config_flags = config_flags.set_vsinh();
        }
        let platform_counter_index = if !CpuInfo::get().has_sscdeleg() {
            sbi::api::pmu::configure_matching_counters(
                counter_index,
                counter_mask,
                config_flags.set_sinh().set_minh(),
                event_type,
                event_data,
            )
        } else {
            // Ensure that skip_match is propogated in the delegated counters case
            self.configure_delegated_matching_counters(
                counter_index,
                counter_mask,
                config_flags,
                event_type,
                event_data,
            )
        }?;
        Ok(platform_counter_index)
    }

    /// Calls either SBI configure_matching_counters(), or the internal implementation depending on
    /// whether Sscdeleg is supported, and performs internal bookkeeping on counter state.
    pub fn configure_matching_counters(
        &mut self,
        counter_index: u64,
        counter_mask: u64,
        config_flags: PmuCounterConfigFlags,
        event_type: PmuEventType,
        event_data: u64,
    ) -> SbiResult<u64> {
        let counter_mask =
            self.get_configurable_counter_range(counter_index, counter_mask, config_flags)?;
        let platform_counter_index = self.configure_matching_counters_internal(
            counter_index,
            counter_mask,
            config_flags,
            event_type,
            event_data,
        )?;
        self.update_configured_counter(
            platform_counter_index,
            config_flags,
            event_type,
            event_data,
        )?;
        Ok(platform_counter_index)
    }

    fn start_counters_internal(
        &mut self,
        counter_index: u64,
        counter_mask: u64,
        start_flags: PmuCounterStartFlags,
        initial_value: u64,
    ) -> SbiResult<()> {
        if !CpuInfo::get().has_sscdeleg() {
            sbi::api::pmu::start_counters(counter_index, counter_mask, start_flags, initial_value)
        } else {
            self.start_delegated_counters(counter_index, counter_mask, start_flags, initial_value)
        }
    }

    /// Calls the SBI start_counters() and performs internal bookkeeping on counter state.
    fn start_delegated_counters(
        &self,
        counter_index: u64,
        counter_mask: u64,
        start_flags: PmuCounterStartFlags,
        initial_value: u64,
    ) -> SbiResult<()> {
        let bitmask_iter = CounterMaskIter::new(counter_index, counter_mask);
        let mut uninhibit_mask = 0u64;
        for index in bitmask_iter {
            if matches!(self.counter_state[index], Configured(_)) {
                let index = index as u64;
                if start_flags.is_init_value() {
                    CSR.scounterselect.set(index);
                    CSR.shpmcounter.set(initial_value);
                }
                uninhibit_mask |= 1 << index;
            }
        }
        if uninhibit_mask != 0 {
            CSR.scountinhibit.read_and_clear_bits(uninhibit_mask);
        }
        Ok(())
    }

    /// Calls either SBI start_counters(), or the internal implementation depending on
    /// whether Sscdeleg is supported, and performs internal bookkeeping on counter state.
    pub fn start_counters(
        &mut self,
        counter_index: u64,
        counter_mask: u64,
        start_flags: PmuCounterStartFlags,
        initial_value: u64,
    ) -> SbiResult<()> {
        let counter_mask = self.get_startable_counter_range(counter_index, counter_mask)?;
        let result =
            self.start_counters_internal(counter_index, counter_mask, start_flags, initial_value);
        // Special case "already started" to handle counters that are autostarted following configuration.
        // Examples of such counters include the legacy timer and insret.
        if result.is_ok() || matches!(result, Err(SbiError::AlreadyStarted)) {
            self.update_started_counters(counter_index, counter_mask);
        }
        result
    }

    fn stop_delegated_counters(
        &self,
        counter_index: u64,
        counter_mask: u64,
        stop_flags: PmuCounterStopFlags,
    ) -> SbiResult<()> {
        let bitmask_iter = CounterMaskIter::new(counter_index, counter_mask);
        let mut inhibit_mask = 0u64;
        for index in bitmask_iter {
            if matches!(self.counter_state[index], Started(_)) {
                let index = index as u64;
                inhibit_mask |= 1 << index;
                if stop_flags.is_reset_flag() {
                    CSR.scounterselect.set(index);
                    CSR.shpmevent.set(0);
                }
            }
        }
        if inhibit_mask != 0 {
            CSR.scountinhibit.read_and_set_bits(inhibit_mask);
        }
        Ok(())
    }

    pub fn stop_counters_internal(
        &mut self,
        counter_index: u64,
        counter_mask: u64,
        stop_flags: PmuCounterStopFlags,
    ) -> SbiResult<()> {
        if !CpuInfo::get().has_sscdeleg() {
            sbi::api::pmu::stop_counters(counter_index, counter_mask, stop_flags)
        } else {
            self.stop_delegated_counters(counter_index, counter_mask, stop_flags)
        }
    }

    /// Calls either SBI stop_counters(), or the internal implementation depending on
    /// whether Sscdeleg is supported, and performs internal bookkeeping on counter state.
    pub fn stop_counters(
        &mut self,
        counter_index: u64,
        counter_mask: u64,
        stop_flags: PmuCounterStopFlags,
    ) -> SbiResult<()> {
        let counter_mask = self.get_stoppable_counter_range(counter_index, counter_mask)?;
        let result = self.stop_counters_internal(counter_index, counter_mask, stop_flags);
        // Special case "already stopped" to handle counters that can be reset following a stop
        if result.is_ok()
            || (matches!(result, Err(SbiError::AlreadyStopped)) && stop_flags.is_reset_flag())
        {
            self.update_stopped_counters(counter_index, counter_mask, stop_flags);
        }
        result
    }
}
