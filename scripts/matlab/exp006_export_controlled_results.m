function exp006_export_controlled_results(result_root, scenarios_csv, output_csv, metadata_json)
%EXP006_EXPORT_CONTROLLED_RESULTS Export labeled official-simulator results.

arguments
    result_root (1, 1) string
    scenarios_csv (1, 1) string
    output_csv (1, 1) string
    metadata_json (1, 1) string
end

if ~isfolder(result_root)
    error("EXP006:MissingResults", "Simulator result directory not found: %s", result_root);
end
scenarios = readtable(scenarios_csv, VariableNamingRule="preserve", TextType="string");

overview_files = dir(fullfile(result_root, "**", "*_results_overview.mat"));
if isempty(overview_files)
    error("EXP006:MissingOverview", "No result overview was found under %s.", result_root);
end
overview_parts = cell(numel(overview_files), 1);
for index = 1:numel(overview_files)
    loaded = load(fullfile(overview_files(index).folder, overview_files(index).name), ...
        "Results_Overview");
    overview_parts{index} = loaded.Results_Overview;
end
overview = vertcat(overview_parts{:});

signal_files = dir(fullfile(result_root, "**", "*_acceleration_signal_*.mat"));
if height(overview) ~= numel(signal_files)
    error("EXP006:ResultCount", "Overview has %d rows but %d signal files exist.", ...
        height(overview), numel(signal_files));
end

records = {};
run_snapshot_counts = zeros(numel(signal_files), 1);
families = strings(numel(signal_files), 1);
for file_index = 1:numel(signal_files)
    path = fullfile(signal_files(file_index).folder, signal_files(file_index).name);
    loaded = load(path, "BP", "OC", "Sim", "accSignal");
    scenario_id = string(loaded.Sim.simulation_name);
    scenario_index = find(string(scenarios.scenario_id) == scenario_id);
    if numel(scenario_index) ~= 1
        error("EXP006:ScenarioIdentity", "Could not uniquely match scenario %s.", scenario_id);
    end
    overview_index = find(string(overview.simulation_name) == scenario_id);
    if numel(overview_index) ~= 1
        error("EXP006:OverviewIdentity", "Could not uniquely match overview %s.", scenario_id);
    end

    degradation_time = local_unbox_numeric(overview.degradation_time(overview_index, :));
    degradation_value = local_unbox_numeric(overview.degradation(overview_index, :));
    if height(loaded.accSignal) ~= numel(degradation_time) || ...
            numel(degradation_value) ~= numel(degradation_time)
        error("EXP006:Alignment", "Truth/signal length mismatch for %s.", scenario_id);
    end
    terminal_minutes = degradation_time(end);
    duration_minutes = terminal_minutes - degradation_time(1);
    run_snapshot_counts(file_index) = height(loaded.accSignal);
    families(file_index) = string(loaded.Sim.deg_target);
    publication_split = string(scenarios.publication_split(scenario_index));
    condition_id = string(scenarios.condition_id(scenario_index));
    fault_location = local_scalar_text(overview.fault_location(overview_index, :));

    for measurement_index = 1:height(loaded.accSignal)
        acceleration = loaded.accSignal.Acceleration{measurement_index};
        feature = exp006_signal_features(acceleration, loaded.OC.f_sampling);
        elapsed_minutes = degradation_time(measurement_index) - degradation_time(1);
        rul_minutes = terminal_minutes - degradation_time(measurement_index);
        record = struct();
        record.dataset = "exp006_controlled_synthetic";
        record.run_id = scenario_id;
        record.official_partition = publication_split;
        record.condition_id = condition_id;
        record.simulation_number = double(loaded.Sim.simulation_number);
        record.sample_index = measurement_index - 1;
        record.measurement = double(loaded.accSignal.Measurement{measurement_index});
        record.elapsed_minutes = elapsed_minutes;
        record.elapsed_seconds = elapsed_minutes * 60.0;
        record.rul_minutes = rul_minutes;
        record.rul_norm = rul_minutes / max(duration_minutes, eps);
        record.truth_available = true;
        record.degradation_family = string(loaded.Sim.deg_target);
        record.fault_location = fault_location;
        record.degradation_value = degradation_value(measurement_index);
        record.sampling_hz = double(loaded.OC.f_sampling);
        record.load_n = double(loaded.OC.real_load);
        record.load_mean_n = double(loaded.OC.mean_belastung);
        record.load_std_n = double(loaded.OC.std_belastung);
        record.speed_rpm = double(loaded.OC.rpm);
        record.a_iso = double(loaded.OC.a_ISO);
        record.bearing_name = string(loaded.BP.name);
        record.roller_count = double(loaded.BP.n_roller);
        record.roller_diameter_mm = double(loaded.BP.d);
        record.pitch_diameter_mm = double(loaded.BP.D);
        record.contact_angle_deg = double(loaded.BP.alpha);
        record.dynamic_load_rating_n = double(loaded.BP.C);
        record.life_exponent = double(loaded.BP.p);
        record.weibull_shape = double(loaded.BP.b_form);
        record.simulation_name = scenario_id;
        record.sdof_stiffness_n_per_m = double(loaded.Sim.k);
        record.sdof_damping_ns_per_m = double(loaded.Sim.c);
        record.sdof_mass_kg = double(loaded.Sim.m);
        record.gamma_alpha = local_optional_numeric( ...
            loaded.Sim, "gamma_process_alpha", NaN);
        record.gamma_beta = local_optional_numeric( ...
            loaded.Sim, "gamma_process_beta", NaN);
        record.slip_mean = double(loaded.Sim.constSlip);
        record.snr_db = double(loaded.Sim.SNR_dB);
        feature_names = fieldnames(feature);
        for feature_index = 1:numel(feature_names)
            name = feature_names{feature_index};
            record.(name) = feature.(name);
        end
        records{end + 1, 1} = record; %#ok<AGROW>
    end
end

feature_table = struct2table(vertcat(records{:}));
feature_table = sortrows(feature_table, ["official_partition", "run_id", "sample_index"]);
local_atomic_writetable(feature_table, output_csv);

metadata = struct();
metadata.schema_version = 1;
metadata.experiment_id = "EXP-006";
metadata.dataset_id = "exp006_controlled_synthetic";
metadata.matlab_version = string(version);
metadata.exported_at_utc = string(datetime("now", "TimeZone", "UTC", ...
    "Format", "yyyy-MM-dd'T'HH:mm:ss'Z'"));
metadata.run_count = numel(signal_files);
metadata.snapshot_count = height(feature_table);
metadata.minimum_snapshots_per_run = min(run_snapshot_counts);
metadata.maximum_snapshots_per_run = max(run_snapshot_counts);
metadata.progression_families = sort(unique(families))';
metadata.degradation_family_disclosed = true;
metadata.fault_location_disclosed = true;
metadata.physics_truth_available = true;
metadata.output_columns = string(feature_table.Properties.VariableNames);
local_atomic_write_json(metadata, metadata_json);
end


function value = local_unbox_numeric(table_value)
if iscell(table_value)
    value = double(table_value{1}(:));
else
    value = double(table_value(:));
end
end


function value = local_scalar_text(table_value)
if iscell(table_value)
    value = string(table_value{1});
else
    value = string(table_value(1));
end
end


function value = local_optional_numeric(structure, field_name, default_value)
if isfield(structure, field_name)
    value = double(structure.(field_name));
else
    value = double(default_value);
end
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
    error("EXP006:Write", "Could not open temporary metadata file.");
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
