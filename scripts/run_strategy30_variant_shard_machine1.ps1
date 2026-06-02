param(
    [string]$RunGroupId = $env:RUN_GROUP_ID,
    [string]$MachineId = $env:MACHINE_ID,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunGroupId)) {
    throw "Set RunGroupId, for example: .\scripts\run_strategy30_variant_shard_machine1.ps1 -RunGroupId sensys_strategy30_20260602"
}

if ([string]::IsNullOrWhiteSpace($MachineId)) {
    $MachineId = "1"
}

$argsList = @(
    "scripts/run_strategy_30task_split.py",
    "--run_group_id", $RunGroupId,
    "--machine_id", $MachineId,
    "--task_shard_id", "ALL",
    "--max_cases", "30",
    "--variants", "B0_BASELINE_RERUN,S1_BFS_BUDGET12,S3_BEAM_BUDGET12,S5_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12"
)

if ($DryRun) {
    $argsList += "--dry_run"
}

& python @argsList
exit $LASTEXITCODE
