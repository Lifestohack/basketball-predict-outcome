clear
readpath = 'dataset\matlab';
savepath = 'dataset\trajectory';
start(readpath, savepath)

function start(readpath, savepath)
    rootdir = dir(fullfile(readpath));
    step = 0;
    total = length(dir(fullfile(readpath, "**/*.png")));
    for n = 1 : length(rootdir)
         %img_path = fullfile(rootdir(n).folder,rootdir(n).name)
         if rootdir(n).name ~= "."
            if rootdir(n).name ~= ".."
                trainorvalid = fullfile(rootdir(n).folder, rootdir(n).name);
                trainorvaliddir = dir(trainorvalid);
                for i = 1 : length(trainorvaliddir)
                     if trainorvaliddir(i).name ~= "."
                        if trainorvaliddir(i).name ~= ".."
                            if contains(trainorvaliddir(i).name, "hit") | contains(trainorvaliddir(i).name, "miss")
                                hitormiss = fullfile(trainorvaliddir(i).folder, trainorvaliddir(i).name);
                                hitormissddir = dir(hitormiss);
                                for j = 1 : length(hitormissddir)
                                    if hitormissddir(j).name ~= "."
                                        if hitormissddir(j).name ~= ".."
                                            sample = fullfile(hitormissddir(j).folder, hitormissddir(j).name);
                                            sampleddir = dir(sample);
                                            for k = 1 : length(sampleddir)
                                                if sampleddir(k).name ~= "."
                                                    if sampleddir(k).name ~= ".."
                                                        view1 = fullfile(sampleddir(k).folder, sampleddir(k).name, "**/*.png");
                                                        view1dir = dir(view1);
                                                        view(view1dir,readpath, savepath)
                                                        step = step + 99;
                                                        disp((step/total)*100);   
                                                    end
                                                end
                                            end
                                
                                        end
                                    end
                                end
                            else
                                %validation
                                sample = fullfile(trainorvaliddir(i).folder, trainorvaliddir(i).name);
                                sampleddir = dir(sample);
                                for k = 1 : length(sampleddir)
                                      if sampleddir(k).name ~= "."
                                         if sampleddir(k).name ~= ".."
                                              view1 = fullfile(sampleddir(k).folder, sampleddir(k).name, "**/*.png");
                                              view1dir = dir(view1);
                                              view(view1dir, readpath, savepath);
                                              step = step + 99;
                                              disp((step/total)*100);   
                                          end
                                      end
                                end
                            end
                        end
                     end
                end
            end
         end
    end
end

function view(view1dir, readpath, savepath)
    datatocsv = [];
    for l = 1 : length(view1dir)
        image = fullfile(view1dir(l).folder, view1dir(l).name );
        [center, radius] = getcirclecoordinates(image);
        if ~isempty(center)
            datatocsv = [datatocsv;[l, center, radius]];
        end
    end
    pos = process(datatocsv);
    [filepath,name,ext] = fileparts(image);
    csvsavepath = strrep(image,strcat(name, ext),'trajectory.csv');
    csvsavepath = strrep(csvsavepath,readpath,savepath);
    [filepath,name,ext] = fileparts(csvsavepath);
    if ~exist(filepath, 'dir')
        mkdir(filepath);
    end
    csvwrite(csvsavepath,pos);
end

function [centersBright,radiiBright] = getcirclecoordinates(path)
    if contains(path, "view1")
       radius = [13 20];
       sense = 0.95;
       edge = 0.1;
    else
       radius = [15 40];    
       sense = 0.95;
       edge = 0.2;
    end
    rgb = imread(path);
    %rgb = rgb2gray(rgb);
    %imshow(rgb)
    [centersBright,radiiBright,metricBright] = imfindcircles(rgb,radius, ...
        'ObjectPolarity','bright','Sensitivity',sense,'EdgeThreshold',edge);
    
    if contains(path, "view1")
        %don't need circle in the subject body if it is already thrown. 
        % it is probably error
        for i = length(radiiBright):-1:1
            if centersBright(i,1) >= 750 & centersBright(i,2) >= 320
                radiiBright(i) = [];
                centersBright(i,:) = [];
            end
        end
    else
       for i = length(radiiBright):-1:1
            if centersBright(i,2) >= 600
                radiiBright(i) = [];
                centersBright(i,:) = [];
            end
        end
    end  
    if length(radiiBright) > 1
        if contains(path, "view1")
            [M,index_radiiBright] = max(radiiBright);
        else
            [M,index_radiiBright] = max(radiiBright);
        end

        for i = length(radiiBright):-1:1
            if i~=index_radiiBright
                radiiBright(i) = [];
                centersBright(i,:) = [];
            end
        end
    end
    %hBright = viscircles(centersBright, radiiBright,'Color','r');
    %pause(0.01)
end

function position = process(position)
    newposition = [];
    for n = 1:length(position)
        if n == 1
            continue
        end
        points = position(n,1) - position(n-1,1) -1 ;
        if points==0
            continue
        else
            p = linspace(position(n-1, 1), position(n,1), points + 2);
            x = linspace(position(n-1, 2), position(n,2), points + 2);
            y = linspace(position(n-1, 3), position(n,3), points + 2);
            r = linspace(position(n-1, 4), position(n,4), points + 2);
            p(1) = [];p(end) = [];
            x(1) = [];x(end) = [];
            y(1) = [];y(end) = [];
            r(1) = [];r(end) = [];
            for i=1:length(p)
                newposition = [newposition;[p(i), x(i), y(i), r(i)]];
            end
        end
    end
    
    %copy first frames values from the first available values
    for i=position(1,1)-1:-1:1
        newposition = [newposition;[i, position(1,2), position(1,3), position(1,4)]];
    end
    %copy first frames values from the first available values
    for i=position(end,1)+1:99
        newposition = [newposition;[i, position(end,2), position(end,3), position(end,4)]];
    end
    position = cat(1, position, newposition);
    position = sortrows(position,1);
end