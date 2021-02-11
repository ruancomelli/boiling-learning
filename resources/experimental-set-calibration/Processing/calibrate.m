% rpath:= relative path
data_rpath = fullfile('..', 'Calibration', 'Calibration Data');
data_fpath = fullfile(cd, data_rpath);

listing = dir(data_fpath);

row_offset = 23;

for i = 1:length(listing)
    file_struct = listing(i);
    if ~file_struct.isdir
        filename = file_struct.name;
        file_fpath = fullfile(data_fpath, filename);
        
        temperature_cell = regexp(filename, '([0123456789]+)_([0123456789]+)C', 'tokens');
        temperature_cell = {temperature_cell{1}(1), temperature_cell{1}(2)};
        reference_temperature = str2double(temperature_cell{1}{1}) + 0.01*str2double(temperature_cell{2}{1});
        
        table = readtable(file_fpath, 'delimiter', '\t', 'HeaderLines', row_offset);
    end
end