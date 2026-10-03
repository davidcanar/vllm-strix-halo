#!/bin/bash
# gpucap.sh <MHz|auto> : cap (or restore) the iGPU max SCLK on this box (non-persistent, reverts on reboot)
set -e
D=$(for c in /sys/class/drm/card*/device; do [ -e $c/pp_od_clk_voltage ] && echo $c && break; done)
if [ "$1" = auto ]; then
  echo r | sudo tee $D/pp_od_clk_voltage >/dev/null; echo c | sudo tee $D/pp_od_clk_voltage >/dev/null
  echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null
else
  echo manual | sudo tee $D/power_dpm_force_performance_level >/dev/null
  echo "s 1 $1" | sudo tee $D/pp_od_clk_voltage >/dev/null; echo c | sudo tee $D/pp_od_clk_voltage >/dev/null
fi
echo "$(hostname -s) $(cat $D/power_dpm_force_performance_level) $(grep -A2 OD_SCLK $D/pp_od_clk_voltage | tail -1 | tr -s ' ')"
