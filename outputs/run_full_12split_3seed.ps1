Set-Location 'X:\thesis-work'
& uv run thesis-work run --seed-repeats 3 *> 'X:\thesis-work\outputs\full_run_12split_3seed.log'
$LASTEXITCODE | Set-Content -Path 'X:\thesis-work\outputs\full_run_12split_3seed.exitcode' -Encoding ASCII
