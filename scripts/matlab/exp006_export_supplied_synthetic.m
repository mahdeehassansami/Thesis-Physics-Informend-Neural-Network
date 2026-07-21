function exp006_export_supplied_synthetic(input_mat, output_csv, metadata_json)
%EXP006_EXPORT_SUPPLIED_SYNTHETIC Export the supplied MATLAB table safely.
%
% Raw Data.mat is read-only. The function writes a compact, standardized
% feature table and a metadata JSON into a separate derived-data directory.

arguments
    input_mat (1, 1) string
    output_csv (1, 1) string
    metadata_json (1, 1) string
end

if ~isfile(input_mat)
    error("EXP006:MissingInput", "Input file does not exist: %s", input_mat);
end

contents = whos("-file", input_mat);
available = string({contents.name});
expected_variables = ["Train_Data", "Test_Data"];
if ~all(ismember(expected_variables, available))
    error("EXP006:Schema", "Data.mat must contain Train_Data and Test_Data.");
end

loaded = load(input_mat, "Train_Data", "Test_Data");
expected_columns = ["no", "degradation_time", "signal", "Bearing", ...
    "OperatingConditions", "SimulationDetails"];
signal_columns = ["Measurement", "Time", "Acceleration"];
partition_variables = ["Train_Data", "Test_Data"];
partition_labels = ["train", "test"];
expected_runs = [28, 12];

total_snapshots = 0;
snapshot_counts = zeros(1, numel(partition_variables));
run_snapshot_counts = [];
for partition_index = 1:numel(partition_variables)
    source = loaded.(partition_variables(partition_index));
    if height(source) ~= expected_runs(partition_index)
        error("EXP006:RunCount", "%s contains %d rather than %d runs.", ...
            partition_variables(partition_index), height(source), ...
            expected_runs(partition_index));
    end
    if ~all(ismember(expected_columns, string(source.Properties.VariableNames)))
        error("EXP006:Schema", "%s has unexpected columns.", ...
            partition_variables(partition_index));
    end
    counts = cellfun(@numel, source.degradation_time);
    run_snapshot_counts = [run_snapshot_counts; counts(:)]; %#ok<AGROW>
    snapshot_counts(partition_index) = sum(counts);
    total_snapshots = total_snapshots + sum(counts);
end

records = cell(total_snapshots, 1);
record_index = 0;
for partition_index = 1:numel(partition_variables)
    source = loaded.(partition_variables(partition_index));
    partition = partition_labels(partition_index);
    for run_index = 1:height(source)
        run_number = double(source.no(run_index));
        run_id = sprintf("synthetic_%s_%03d", partition, run_number);
        times_minutes = double(source.degradation_time{run_index}(:));
        signals = source.signal{run_index};
        bearing = source.Bearing{run_index};
        operating = source.OperatingConditions{run_index};
        simulation = source.SimulationDetails{run_index};

        if ~all(ismember(signal_columns, string(signals.Properties.VariableNames)))
            error("EXP006:SignalSchema", "Unexpected signal columns for %s.", run_id);
        end
        if height(signals) ~= numel(times_minutes)
            error("EXP006:Alignment", "Signal/time length mismatch for %s.", run_id);
        end
        if any(diff(times_minutes) <= 0)
            error("EXP006:TimeOrder", "Degradation time is not strictly increasing for %s.", run_id);
        end

        terminal_minutes = times_minutes(end);
        duration_minutes = terminal_minutes - times_minutes(1);
        sampling_hz = double(operating.f_sampling);
        for measurement_index = 1:height(signals)
            acceleration = signals.Acceleration{measurement_index};
            time_axis = signals.Time{measurement_index};
            if numel(acceleration) ~= numel(time_axis)
                error("EXP006:SignalAlignment", "Acceleration/time mismatch for %s.", run_id);
            end
            feature = exp006_signal_features(acceleration, sampling_hz);
            elapsed_minutes = times_minutes(measurement_index) - times_minutes(1);
            rul_minutes = terminal_minutes - times_minutes(measurement_index);
            record_index = record_index + 1;
            records{record_index} = local_record( ...
                "synthetic_varying_degradation_v2", run_id, partition, ...
                run_number, measurement_index - 1, elapsed_minutes, rul_minutes, ...
                duration_minutes, double(signals.Measurement{measurement_index}), ...
                sampling_hz, bearing, operating, simulation, feature);
        end
    end
end

if record_index ~= total_snapshots
    error("EXP006:InternalCount", "Exported %d rather than %d snapshots.", ...
        record_index, total_snapshots);
end

feature_table = struct2table(vertcat(records{:}));
local_atomic_writetable(feature_table, output_csv);

metadata = struct();
metadata.schema_version = 1;
metadata.experiment_id = "EXP-006";
metadata.dataset_id = "synthetic_varying_degradation_v2";
metadata.source_file = "Datasets/Bearings_with_Varying_Degradation_Behaviors_v2/Data.mat";
metadata.matlab_version = string(version);
metadata.exported_at_utc = string(datetime("now", "TimeZone", "UTC", ...
    "Format", "yyyy-MM-dd'T'HH:mm:ss'Z'"));
metadata.run_counts = struct("train", 28, "test", 12, "total", 40);
metadata.snapshot_counts = struct("train", snapshot_counts(1), ...
    "test", snapshot_counts(2), "total", total_snapshots, ...
    "minimum_per_run", min(run_snapshot_counts), ...
    "maximum_per_run", max(run_snapshot_counts));
metadata.degradation_family_disclosed = false;
metadata.fault_location_disclosed = false;
metadata.physics_truth_available = false;
metadata.identifiability_note = ...
    "The dataset documentation intentionally withholds the degradation " + ...
    "function and fault type. This cache supports RUL evaluation but cannot " + ...
    "supply known-valid physics labels.";
metadata.feature_columns = string(fieldnames(exp006_signal_features( ...
    loaded.Train_Data.signal{1}.Acceleration{1}, ...
    loaded.Train_Data.OperatingConditions{1}.f_sampling)))';
metadata.output_columns = string(feature_table.Properties.VariableNames);
local_atomic_write_json(metadata, metadata_json);
end


function record = local_record(dataset_id, run_id, partition, run_number, ...
        sample_index, elapsed_minutes, rul_minutes, duration_minutes, measurement, ...
        sampling_hz, bearing, operating, simulation, feature)
record = struct();
record.dataset = string(dataset_id);
record.run_id = string(run_id);
record.official_partition = string(partition);
record.simulation_number = run_number;
record.sample_index = sample_index;
record.measurement = measurement;
record.elapsed_minutes = elapsed_minutes;
record.elapsed_seconds = elapsed_minutes * 60.0;
record.rul_minutes = rul_minutes;
record.rul_norm = rul_minutes / max(duration_minutes, eps);
record.truth_available = false;
record.degradation_family = "";
record.fault_location = "";
record.degradation_value = NaN;
record.sampling_hz = sampling_hz;
record.load_n = double(operating.mean_belastung);
record.load_std_n = double(operating.std_belastung);
record.speed_rpm = double(operating.rpm);
record.a_iso = double(operating.a_ISO);
record.bearing_name = string(bearing.name);
record.roller_count = double(bearing.n_roller);
record.roller_diameter_mm = double(bearing.d);
record.pitch_diameter_mm = double(bearing.D);
record.contact_angle_deg = double(bearing.alpha);
record.dynamic_load_rating_n = double(bearing.C);
record.life_exponent = double(bearing.p);
record.weibull_shape = double(bearing.b_form);
record.simulation_name = string(simulation.simulation_name);
record.sdof_stiffness_n_per_m = double(simulation.k);
record.sdof_damping_ns_per_m = double(simulation.c);
record.sdof_mass_kg = double(simulation.m);
feature_names = fieldnames(feature);
for index = 1:numel(feature_names)
    name = feature_names{index};
    record.(name) = feature.(name);
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
