function result_directory = exp006_run_controlled_simulator( ...
        simulator_archive, scenarios_csv, output_root, seed, smoke_only)
%EXP006_RUN_CONTROLLED_SIMULATOR Run the official CC BY 4.0 MATLAB model.
%
% The scenario CSV is the authoritative parameter source. A runtime XLSX is
% generated only because the upstream P-code expects the provided workbook
% contract. Raw simulator results are written to a derived-data directory.

arguments
    simulator_archive (1, 1) string
    scenarios_csv (1, 1) string
    output_root (1, 1) string
    seed (1, 1) double {mustBeInteger, mustBeNonnegative}
    smoke_only (1, 1) logical = false
end

if ~isfile(simulator_archive)
    error("EXP006:MissingSimulator", "Simulator archive not found: %s", simulator_archive);
end
if ~isfile(scenarios_csv)
    error("EXP006:MissingScenarios", "Scenario file not found: %s", scenarios_csv);
end

required_licenses = ["Signal_Toolbox", "Statistics_Toolbox", "Communication_Toolbox"];
for index = 1:numel(required_licenses)
    if ~license("test", required_licenses(index))
        error("EXP006:MissingToolbox", "Required MATLAB toolbox license is unavailable: %s", ...
            required_licenses(index));
    end
end

scenarios = readtable(scenarios_csv, VariableNamingRule="preserve", TextType="string");
official_columns = ["number", "simulation_name", "BP_name", "BP_d", "BP_D", ...
    "BP_n_roller", "BP_alpha", "BP_C", "BP_p", "BP_b_form", ...
    "OC_load_mean", "OC_load_std", "OC_a_ISO", "OC_f_set", "OC_f_d", ...
    "OC_f_m", "OC_f_sampling", "OC_T_measure_deg", "OC_T_measure_acc", ...
    "SD_degradation_progression", "SD_gamma_process_alpha", ...
    "SD_gamma_process_beta", "SD_slip_mean", "SD_SDOF_m", "SD_SDOF_k", ...
    "SD_SDOF_c", "SD_SNR"];
if ~all(ismember(official_columns, string(scenarios.Properties.VariableNames)))
    error("EXP006:ScenarioSchema", "Scenario CSV does not contain the official columns.");
end
if smoke_only
    scenarios = scenarios(1, :);
end

if ~isfolder(output_root)
    mkdir(output_root);
end
working_directory = string(tempname(output_root));
mkdir(working_directory);
cleanup = onCleanup(@() local_remove_working_directory(working_directory));
unzip(simulator_archive, working_directory);

parameter_table = scenarios(:, cellstr(official_columns));
parameter_workbook = fullfile(working_directory, "EXP006_Simulation_Parameters.xlsx");
writetable(parameter_table, parameter_workbook);

previous_directory = string(pwd);
directory_cleanup = onCleanup(@() cd(previous_directory));
cd(working_directory);
rng(seed, "twister");
saveSignal = 1; %#ok<NASGU> Upstream P-code reads workspace variables.
nameParameterList = "EXP006_Simulation_Parameters.xlsx"; %#ok<NASGU>
nameFolderResults = "EXP006_Controlled_Results"; %#ok<NASGU>
started_at = datetime("now", "TimeZone", "UTC");
Main_Setup_Simulation;
finished_at = datetime("now", "TimeZone", "UTC");

created = dir(fullfile(working_directory, "*EXP006_Controlled_Results"));
created = created([created.isdir]);
if numel(created) ~= 1
    error("EXP006:SimulatorOutput", "Expected one result directory, found %d.", numel(created));
end

run_label = "exp006_controlled_seed_" + string(seed);
if smoke_only
    run_label = run_label + "_smoke";
end
result_directory = fullfile(output_root, run_label);
if isfolder(result_directory)
    error("EXP006:ExistingOutput", "Refusing to overwrite existing simulator output: %s", ...
        result_directory);
end
movefile(fullfile(created(1).folder, created(1).name), result_directory);

run_metadata = struct();
run_metadata.schema_version = 1;
run_metadata.experiment_id = "EXP-006";
run_metadata.seed = seed;
run_metadata.rng = "twister";
run_metadata.smoke_only = smoke_only;
run_metadata.scenario_count = height(scenarios);
run_metadata.scenario_ids = string(scenarios.scenario_id)';
run_metadata.matlab_version = string(version);
run_metadata.started_at_utc = string(started_at, "yyyy-MM-dd'T'HH:mm:ss'Z'");
run_metadata.finished_at_utc = string(finished_at, "yyyy-MM-dd'T'HH:mm:ss'Z'");
run_metadata.elapsed_seconds = seconds(finished_at - started_at);
run_metadata.simulator_license = "CC BY 4.0";
run_metadata.simulator_citation = ...
    "Mauthe, Hagmeyer, and Zeiler (2025), DOI " + ...
    "10.3850/978-981-94-3281-3_ESREL-SRA-E2025-P8028-cd";
local_write_json(run_metadata, fullfile(result_directory, "exp006_simulator_run.json"));

clear directory_cleanup;
clear cleanup;
local_remove_working_directory(working_directory);
end


function local_write_json(value, destination)
handle = fopen(destination, "w");
if handle < 0
    error("EXP006:Write", "Could not write simulator metadata: %s", destination);
end
cleanup = onCleanup(@() fclose(handle));
fprintf(handle, "%s\n", jsonencode(value, PrettyPrint=true));
clear cleanup;
end


function local_remove_working_directory(path)
if isfolder(path)
    resolved = string(java.io.File(char(path)).getCanonicalPath());
    if contains(resolved, "exp006", IgnoreCase=true)
        rmdir(path, "s");
    else
        warning("EXP006:Cleanup", "Refusing to remove unexpected temporary path: %s", resolved);
    end
end
end
