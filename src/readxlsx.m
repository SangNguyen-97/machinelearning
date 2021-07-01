%%% Code to read from xlsx files (R2021a)
%%% xlsx files are stored at "./data/..."

% Constants
metauntil = 6; % Number of columns of metadata (metadata takes the first columns)
varsize = 3400; % Number of columns of time data

% Generate name for time sample data columns (features)
varcell = {};
for i = 1:varsize
    b = num2str(i);
    varcell{1,i} = strcat('t',b);
end

% Read data as table from xlsx
t_all = [];
for obj = ["Object 1" "Object 2" "Object 3" "Object 4" "Object 5"]
    locat = strcat("data/",obj);
    ds = datastore(locat);
    ds.ReadVariableNames = false;
    t = readall(ds);
    t(:,1:metauntil) = [];
    t_all.Properties.VariableNames = varcell; % Rename time data columns (features)
    numrow = size(t,1);
    t.Object(1:numrow,1) = obj; % Add label to data
    t.Object = categorical(t.Object);
    t_all = [t_all;t];
end

