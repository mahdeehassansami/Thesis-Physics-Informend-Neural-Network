function exp007b_export_fresh_test_results( ...
        result_root, scenarios_csv, output_csv, metadata_json)
%EXP007B_EXPORT_FRESH_TEST_RESULTS Export the separately generated 16-run test.

arguments
    result_root (1, 1) string
    scenarios_csv (1, 1) string
    output_csv (1, 1) string
    metadata_json (1, 1) string
end

if ~isfolder(result_root)
    error("EXP007B:MissingResults", "Fresh simulator results not found: %s", result_root);
end
scenarios = readtable(scenarios_csv, VariableNamingRule="preserve", TextType="string");
fresh = scenarios(scenarios.publication_split == "test", :);
if height(fresh) ~= 16 || any(double(fresh.simulator_seed) ~= 920072)
    error("EXP007B:ScenarioDesign", "The fresh test scenario definition changed.");
end

temporary_root = string(tempname(fileparts(output_csv)));
mkdir(temporary_root);
cleanup = onCleanup(@() local_remove_temporary(temporary_root));
temporary_csv = fullfile(temporary_root, "exp006_compatible.csv");
temporary_json = fullfile(temporary_root, "exp006_compatible.json");
exp006_export_controlled_results(result_root, scenarios_csv, temporary_csv, temporary_json);

features = readtable(temporary_csv, VariableNamingRule="preserve", TextType="string");
if numel(unique(features.run_id)) ~= 16 || any(features.official_partition ~= "test")
    error("EXP007B:FreshResultCount", "Fresh export must contain exactly 16 test trajectories.");
end
features.dataset(:) = "exp007b_multicondition_synthetic";
identity = fresh(:, ["scenario_id", "simulator_seed", "sealed_test"]);
identity.Properties.VariableNames{1} = 'run_id';
features = outerjoin(features, identity, Keys="run_id", MergeKeys=true, Type="left");
if any(ismissing(features.simulator_seed)) || any(double(features.simulator_seed) ~= 920072)
    error("EXP007B:Identity", "A fresh trajectory did not match the sealed scenario identity.");
end
features = sortrows(features, ["run_id", "sample_index"]);
local_atomic_writetable(features, output_csv);

metadata = struct();
metadata.schema_version = 1;
metadata.experiment_id = "EXP-007B";
metadata.protocol_version = "0.3.0";
metadata.preregistration_commit = "b8e4b1e18845e7056fd70c6956426483360975f3";
metadata.dataset_id = "exp007b_multicondition_synthetic_fresh_test";
metadata.matlab_version = string(version);
metadata.exported_at_utc = string(datetime("now", "TimeZone", "UTC", ...
    "Format", "yyyy-MM-dd'T'HH:mm:ss'Z'"));
metadata.run_count = 16;
metadata.snapshot_count = height(features);
run_sizes = groupsummary(features, "run_id");
metadata.minimum_snapshots_per_run = min(run_sizes.GroupCount);
metadata.maximum_snapshots_per_run = max(run_sizes.GroupCount);
metadata.progression_families = sort(unique(features.degradation_family))';
metadata.sealed_test_simulator_seed = 920072;
metadata.opened_exp007a_test_seed_excluded = 920071;
metadata.sealed_test_generated_separately = true;
metadata.degradation_family_disclosed = true;
metadata.physics_truth_available = true;
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
    error("EXP007B:Write", "Could not open temporary metadata file.");
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
    if contains(resolved, "run_07b", IgnoreCase=true) || contains(resolved, "tmp", IgnoreCase=true)
        rmdir(path, "s");
    else
        warning("EXP007B:Cleanup", "Refusing unexpected temporary directory: %s", resolved);
    end
end
end

