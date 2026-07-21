function result_directories = exp007a_run_multicondition_simulator( ...
        simulator_archive, scenarios_csv, output_root, smoke_only)
%EXP007A_RUN_MULTICONDITION_SIMULATOR Run the frozen two-seed scenario design.
%
% The tracked CSV is authoritative. Development/validation and sealed test
% scenarios are executed in separate official-simulator invocations using
% their predeclared RNG seeds. Raw outputs are written only below output_root.

arguments
    simulator_archive (1, 1) string
    scenarios_csv (1, 1) string
    output_root (1, 1) string
    smoke_only (1, 1) logical = false
end

if ~isfile(simulator_archive)
    error("EXP007A:MissingSimulator", "Simulator archive not found: %s", simulator_archive);
end
if ~isfile(scenarios_csv)
    error("EXP007A:MissingScenarios", "Scenario file not found: %s", scenarios_csv);
end

required_licenses = ["Signal_Toolbox", "Statistics_Toolbox", "Communication_Toolbox"];
for index = 1:numel(required_licenses)
    if ~license("test", required_licenses(index))
        error("EXP007A:MissingToolbox", "Required MATLAB toolbox license is unavailable: %s", ...
            required_licenses(index));
    end
end

scenarios = readtable(scenarios_csv, VariableNamingRule="preserve", TextType="string");
required_design_columns = ["scenario_id", "publication_split", "simulator_seed", ...
    "sealed_test"];
official_columns = ["number", "simulation_name", "BP_name", "BP_d", "BP_D", ...
    "BP_n_roller", "BP_alpha", "BP_C", "BP_p", "BP_b_form", ...
    "OC_load_mean", "OC_load_std", "OC_a_ISO", "OC_f_set", "OC_f_d", ...
    "OC_f_m", "OC_f_sampling", "OC_T_measure_deg", "OC_T_measure_acc", ...
    "SD_degradation_progression", "SD_gamma_process_alpha", ...
    "SD_gamma_process_beta", "SD_slip_mean", "SD_SDOF_m", "SD_SDOF_k", ...
    "SD_SDOF_c", "SD_SNR"];
if ~all(ismember([required_design_columns, official_columns], ...
        string(scenarios.Properties.VariableNames)))
    error("EXP007A:ScenarioSchema", "Scenario CSV is missing required columns.");
end
if height(scenarios) ~= 96
    error("EXP007A:ScenarioCount", "Expected 96 predeclared scenarios, found %d.", ...
        height(scenarios));
end
if numel(unique(scenarios.scenario_id)) ~= height(scenarios)
    error("EXP007A:ScenarioIdentity", "Scenario identifiers are not unique.");
end

if ~isfolder(output_root)
    mkdir(output_root);
end
seeds = unique(double(scenarios.simulator_seed), "stable");
if ~isequal(sort(seeds(:))', [420071, 920071])
    error("EXP007A:Seeds", "Expected simulator seeds 420071 and 920071.");
end

result_directories = strings(numel(seeds), 1);
for seed_index = 1:numel(seeds)
    seed = seeds(seed_index);
    selected = scenarios(double(scenarios.simulator_seed) == seed, :);
    if seed == 920071 && ~all(lower(selected.sealed_test) == "true")
        error("EXP007A:Seal", "The sealed-test seed contains non-test scenarios.");
    end
    if seed == 420071 && any(lower(selected.sealed_test) == "true")
        error("EXP007A:Seal", "Development seed contains sealed-test scenarios.");
    end
    if smoke_only
        selected = selected(1, :);
    end
    result_directories(seed_index) = local_run_seed( ...
        simulator_archive, selected, output_root, seed, official_columns, smoke_only);
end
end


function result_directory = local_run_seed( ...
        simulator_archive, scenarios, output_root, seed, official_columns, smoke_only)
working_directory = string(tempname(output_root));
mkdir(working_directory);
cleanup = onCleanup(@() local_remove_working_directory(working_directory));
unzip(simulator_archive, working_directory);

parameter_table = scenarios(:, cellstr(official_columns));
parameter_table.number = (1:height(parameter_table))';
parameter_workbook = fullfile(working_directory, ...
    "EXP007A_Simulation_Parameters_" + string(seed) + ".xlsx");
writetable(parameter_table, parameter_workbook);

[created_folder, started_at, finished_at] = local_invoke_upstream( ...
    working_directory, seed);

run_label = "exp007a_seed_" + string(seed);
if smoke_only
    run_label = run_label + "_smoke";
end
result_directory = fullfile(output_root, run_label);
if isfolder(result_directory)
    error("EXP007A:ExistingOutput", "Refusing to overwrite simulator output: %s", ...
        result_directory);
end
movefile(created_folder, result_directory);

metadata = struct();
metadata.schema_version = 1;
metadata.experiment_id = "EXP-007A";
metadata.seed = seed;
metadata.rng = "twister";
metadata.smoke_only = smoke_only;
metadata.scenario_count = height(scenarios);
metadata.scenario_ids = string(scenarios.scenario_id)';
metadata.partitions = unique(string(scenarios.publication_split))';
metadata.sealed_test = all(lower(scenarios.sealed_test) == "true");
metadata.matlab_version = string(version);
metadata.started_at_utc = string(started_at, "yyyy-MM-dd'T'HH:mm:ss'Z'");
metadata.finished_at_utc = string(finished_at, "yyyy-MM-dd'T'HH:mm:ss'Z'");
metadata.elapsed_seconds = seconds(finished_at - started_at);
metadata.simulator_license = "CC BY 4.0";
metadata.simulator_citation = ...
    "Mauthe, Hagmeyer, and Zeiler (2025), DOI " + ...
    "10.3850/978-981-94-3281-3_ESREL-SRA-E2025-P8028-cd";
local_write_json(metadata, fullfile(result_directory, "exp007a_simulator_run.json"));

clear cleanup;
local_remove_working_directory(working_directory);
end


function [created_folder, started_at, finished_at] = local_invoke_upstream( ...
        runtime_directory_argument, seed_argument)
% Isolate the upstream script because it mutates variables in its caller workspace.
previous_directory_internal = string(pwd);
directory_cleanup_internal = onCleanup(@() cd(previous_directory_internal));
cd(runtime_directory_argument);
rng(seed_argument, "twister");
saveSignal = 1; %#ok<NASGU> Upstream P-code reads these exact workspace names.
nameParameterList = ...
    "EXP007A_Simulation_Parameters_" + string(seed_argument) + ".xlsx"; %#ok<NASGU>
nameFolderResults = "EXP007A_Results_" + string(seed_argument); %#ok<NASGU>
started_at = datetime("now", "TimeZone", "UTC");
Main_Setup_Simulation;
finished_at = datetime("now", "TimeZone", "UTC");

post_run_directory_internal = string(pwd);
entries_internal = dir(post_run_directory_internal);
entries_internal = entries_internal([entries_internal.isdir]);
entry_names_internal = string({entries_internal.name});
expected_fragment_internal = "EXP007A_Results_" + string(seed_argument);
created_internal = entries_internal(contains(entry_names_internal, expected_fragment_internal));
if numel(created_internal) ~= 1
    error("EXP007A:SimulatorOutput", ...
        "Expected one new seed result directory under %s, found %d. Entries: %s", ...
        post_run_directory_internal, numel(created_internal), ...
        strjoin(entry_names_internal, ", "));
end
created_folder = string(fullfile(created_internal(1).folder, created_internal(1).name));
clear directory_cleanup_internal;
cd(previous_directory_internal);
end


function local_write_json(value, destination)
handle = fopen(destination, "w");
if handle < 0
    error("EXP007A:Write", "Could not write simulator metadata: %s", destination);
end
cleanup = onCleanup(@() fclose(handle));
fprintf(handle, "%s\n", jsonencode(value, PrettyPrint=true));
clear cleanup;
end


function local_remove_working_directory(path)
if isfolder(path)
    resolved = string(java.io.File(char(path)).getCanonicalPath());
    if contains(resolved, "exp007a", IgnoreCase=true)
        rmdir(path, "s");
    else
        warning("EXP007A:Cleanup", "Refusing to remove unexpected temporary path: %s", ...
            resolved);
    end
end
end
