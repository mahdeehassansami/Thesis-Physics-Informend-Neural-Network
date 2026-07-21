function result_directory = exp007b_run_fresh_test_simulator( ...
        simulator_archive, scenarios_csv, output_root, smoke_only)
%EXP007B_RUN_FRESH_TEST_SIMULATOR Generate only the preregistered fresh test.

arguments
    simulator_archive (1, 1) string
    scenarios_csv (1, 1) string
    output_root (1, 1) string
    smoke_only (1, 1) logical = false
end

if ~isfile(simulator_archive)
    error("EXP007B:MissingSimulator", "Simulator archive not found: %s", simulator_archive);
end
if ~isfile(scenarios_csv)
    error("EXP007B:MissingScenarios", "Scenario file not found: %s", scenarios_csv);
end
required_licenses = ["Signal_Toolbox", "Statistics_Toolbox", "Communication_Toolbox"];
for index = 1:numel(required_licenses)
    if ~license("test", required_licenses(index))
        error("EXP007B:MissingToolbox", "Required toolbox is unavailable: %s", ...
            required_licenses(index));
    end
end

scenarios = readtable(scenarios_csv, VariableNamingRule="preserve", TextType="string");
if height(scenarios) ~= 96 || numel(unique(scenarios.scenario_id)) ~= 96
    error("EXP007B:ScenarioDesign", "Expected 96 unique frozen scenarios.");
end
selected = scenarios(scenarios.publication_split == "test", :);
if height(selected) ~= 16 || any(double(selected.simulator_seed) ~= 920072) || ...
        ~all(lower(selected.sealed_test) == "true")
    error("EXP007B:FreshSeal", "Fresh test must contain 16 sealed seed-920072 scenarios.");
end
if any(double(scenarios.simulator_seed(scenarios.publication_split ~= "test")) ~= 420071)
    error("EXP007B:DevelopmentSeed", "Development rows changed from seed 420071.");
end
if smoke_only
    selected = selected(1, :);
end
if ~isfolder(output_root)
    mkdir(output_root);
end

official_columns = ["number", "simulation_name", "BP_name", "BP_d", "BP_D", ...
    "BP_n_roller", "BP_alpha", "BP_C", "BP_p", "BP_b_form", ...
    "OC_load_mean", "OC_load_std", "OC_a_ISO", "OC_f_set", "OC_f_d", ...
    "OC_f_m", "OC_f_sampling", "OC_T_measure_deg", "OC_T_measure_acc", ...
    "SD_degradation_progression", "SD_gamma_process_alpha", ...
    "SD_gamma_process_beta", "SD_slip_mean", "SD_SDOF_m", "SD_SDOF_k", ...
    "SD_SDOF_c", "SD_SNR"];
if ~all(ismember(official_columns, string(selected.Properties.VariableNames)))
    error("EXP007B:ScenarioSchema", "Scenario CSV is missing official simulator columns.");
end

working_directory = string(tempname(output_root));
mkdir(working_directory);
cleanup = onCleanup(@() local_remove_working_directory(working_directory));
unzip(simulator_archive, working_directory);
parameter_table = selected(:, cellstr(official_columns));
parameter_table.number = (1:height(parameter_table))';
parameter_workbook = fullfile(working_directory, "EXP007B_Simulation_Parameters_920072.xlsx");
writetable(parameter_table, parameter_workbook);

[created_folder, started_at, finished_at] = local_invoke_upstream(working_directory);
run_label = "exp007b_seed_920072";
if smoke_only
    run_label = run_label + "_smoke";
end
result_directory = fullfile(output_root, run_label);
if isfolder(result_directory)
    error("EXP007B:ExistingOutput", "Refusing to overwrite fresh simulator output: %s", ...
        result_directory);
end
movefile(created_folder, result_directory);

metadata = struct();
metadata.schema_version = 1;
metadata.experiment_id = "EXP-007B";
metadata.protocol_version = "0.3.0";
metadata.preregistration_commit = "b8e4b1e18845e7056fd70c6956426483360975f3";
metadata.seed = 920072;
metadata.rng = "twister";
metadata.smoke_only = smoke_only;
metadata.scenario_count = height(selected);
metadata.scenario_ids = string(selected.scenario_id)';
metadata.partitions = "test";
metadata.sealed_test = true;
metadata.opened_exp007a_test_seed_excluded = 920071;
metadata.matlab_version = string(version);
metadata.started_at_utc = string(started_at, "yyyy-MM-dd'T'HH:mm:ss'Z'");
metadata.finished_at_utc = string(finished_at, "yyyy-MM-dd'T'HH:mm:ss'Z'");
metadata.elapsed_seconds = seconds(finished_at - started_at);
metadata.simulator_license = "CC BY 4.0";
metadata.simulator_citation = ...
    "Mauthe, Hagmeyer, and Zeiler (2025), DOI " + ...
    "10.3850/978-981-94-3281-3_ESREL-SRA-E2025-P8028-cd";
local_write_json(metadata, fullfile(result_directory, "exp007b_simulator_run.json"));

clear cleanup;
local_remove_working_directory(working_directory);
end


function [created_folder, started_at, finished_at] = local_invoke_upstream(runtime_directory)
previous_directory = string(pwd);
directory_cleanup = onCleanup(@() cd(previous_directory));
cd(runtime_directory);
rng(920072, "twister");
saveSignal = 1; %#ok<NASGU>
nameParameterList = "EXP007B_Simulation_Parameters_920072.xlsx"; %#ok<NASGU>
nameFolderResults = "EXP007B_Results_920072"; %#ok<NASGU>
started_at = datetime("now", "TimeZone", "UTC");
Main_Setup_Simulation;
finished_at = datetime("now", "TimeZone", "UTC");
entries = dir(string(pwd));
entries = entries([entries.isdir]);
names = string({entries.name});
created = entries(contains(names, "EXP007B_Results_920072"));
if numel(created) ~= 1
    error("EXP007B:SimulatorOutput", "Expected one fresh result directory, found %d.", ...
        numel(created));
end
created_folder = string(fullfile(created(1).folder, created(1).name));
clear directory_cleanup;
cd(previous_directory);
end


function local_write_json(value, destination)
handle = fopen(destination, "w");
if handle < 0
    error("EXP007B:Write", "Could not write simulator metadata: %s", destination);
end
cleanup = onCleanup(@() fclose(handle));
fprintf(handle, "%s\n", jsonencode(value, PrettyPrint=true));
clear cleanup;
end


function local_remove_working_directory(path)
if isfolder(path)
    resolved = string(java.io.File(char(path)).getCanonicalPath());
    if contains(resolved, "run_07b", IgnoreCase=true)
        rmdir(path, "s");
    else
        warning("EXP007B:Cleanup", "Refusing unexpected temporary path: %s", resolved);
    end
end
end

