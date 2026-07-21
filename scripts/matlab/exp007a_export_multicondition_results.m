function exp007a_export_multicondition_results( ...
        result_root, scenarios_csv, output_csv, metadata_json)
%EXP007A_EXPORT_MULTICONDITION_RESULTS Export the frozen 96-run benchmark.
%
% Reuses the verified EXP-006 signal/truth alignment exporter, then applies
% EXP-007A identity and simulator-seed/seal metadata. Raw MAT files are read
% only; the output CSV/JSON are derived caches.

arguments
    result_root (1, 1) string
    scenarios_csv (1, 1) string
    output_csv (1, 1) string
    metadata_json (1, 1) string
end

if ~isfolder(result_root)
    error("EXP007A:MissingResults", "Simulator result directory not found: %s", ...
        result_root);
end
scenarios = readtable(scenarios_csv, VariableNamingRule="preserve", TextType="string");
if height(scenarios) ~= 96
    error("EXP007A:ScenarioCount", "Expected 96 scenarios, found %d.", height(scenarios));
end

temporary_root = string(tempname(fileparts(output_csv)));
mkdir(temporary_root);
cleanup = onCleanup(@() local_remove_temporary(temporary_root));
temporary_csv = fullfile(temporary_root, "exp006_compatible.csv");
temporary_json = fullfile(temporary_root, "exp006_compatible.json");
exp006_export_controlled_results(result_root, scenarios_csv, temporary_csv, temporary_json);

features = readtable(temporary_csv, VariableNamingRule="preserve", TextType="string");
features.dataset(:) = "exp007a_multicondition_synthetic";
identity = scenarios(:, ["scenario_id", "simulator_seed", "sealed_test"]);
identity.Properties.VariableNames{1} = "run_id";
features = outerjoin(features, identity, Keys="run_id", MergeKeys=true, Type="left");
if any(ismissing(features.simulator_seed))
    error("EXP007A:Identity", "A derived trajectory did not match its scenario identity.");
end
features = sortrows(features, ["official_partition", "run_id", "sample_index"]);
local_atomic_writetable(features, output_csv);

counts = groupsummary(features, "official_partition", "numel", "run_id");
metadata = struct();
metadata.schema_version = 1;
metadata.experiment_id = "EXP-007A";
metadata.dataset_id = "exp007a_multicondition_synthetic";
metadata.matlab_version = string(version);
metadata.exported_at_utc = string(datetime("now", "TimeZone", "UTC", ...
    "Format", "yyyy-MM-dd'T'HH:mm:ss'Z'"));
metadata.run_count = numel(unique(features.run_id));
metadata.snapshot_count = height(features);
run_sizes = groupsummary(features, "run_id");
metadata.minimum_snapshots_per_run = min(run_sizes.GroupCount);
metadata.maximum_snapshots_per_run = max(run_sizes.GroupCount);
metadata.progression_families = sort(unique(features.degradation_family))';
metadata.development_simulator_seed = 420071;
metadata.sealed_test_simulator_seed = 920071;
metadata.sealed_test_generated_separately = true;
metadata.degradation_family_disclosed = true;
metadata.physics_truth_available = true;
metadata.partition_rows = table2struct(counts);
metadata.output_columns = string(features.Properties.VariableNames);
metadata.source_simulator_license = "CC BY 4.0";
metadata.source_simulator_citation = ...
    "Mauthe, Hagmeyer, and Zeiler (2025), DOI " + ...
    "10.3850/978-981-94-3281-3_ESREL-SRA-E2025-P8028-cd";
local_atomic_write_json(metadata, metadata_json);
clear cleanup;
local_remove_temporary(temporary_root);
end


function local_atomic_writetable(value, destination)
destination = string(destination);
folder = fileparts(destination);
if strlength(folder) > 0 && ~isfolder(folder)
    mkdir(folder);
end
temporary = string(tempname(folder)) + ".csv";
cleanup = onCleanup(@() local_delete_if_present(temporary));
writetable(value, temporary);
movefile(temporary, destination, "f");
clear cleanup;
end


function local_atomic_write_json(value, destination)
destination = string(destination);
folder = fileparts(destination);
if strlength(folder) > 0 && ~isfolder(folder)
    mkdir(folder);
end
temporary = string(tempname(folder)) + ".json";
cleanup = onCleanup(@() local_delete_if_present(temporary));
handle = fopen(temporary, "w");
if handle < 0
    error("EXP007A:Write", "Could not open temporary metadata file.");
end
file_cleanup = onCleanup(@() fclose(handle));
fprintf(handle, "%s\n", jsonencode(value, PrettyPrint=true));
clear file_cleanup;
movefile(temporary, destination, "f");
clear cleanup;
end


function local_delete_if_present(path)
if isfile(path)
    delete(path);
end
end


function local_remove_temporary(path)
if isfolder(path)
    resolved = string(java.io.File(char(path)).getCanonicalPath());
    if contains(resolved, "exp007a", IgnoreCase=true) || contains(resolved, "tmp", IgnoreCase=true)
        rmdir(path, "s");
    else
        warning("EXP007A:Cleanup", "Refusing unexpected temporary directory: %s", resolved);
    end
end
end
