clear

readpath = 'D:\dataset\withoutbackground';
savepath = 'D:\dataset\trajectory';
start(readpath, savepath)


path = 'D:\dataset\withoutbackground\training\hit\0\view2'

rootdir = dir(fullfile(path, "**/*.png"));
for n = 1:length(rootdir)
    path = fullfile(rootdir(n).folder, rootdir(n).name);
    getcirclecoordinates(path)
end



function start(readpath, savepath)
    rootdir = dir(fullfile(readpath));
    step = 0;
    for n = 1 : length(rootdir)
         %img_path = fullfile(rootdir(n).folder,rootdir(n).name)
         if rootdir(n).name ~= "."
            if rootdir(n).name ~= ".."
                trainorvalid = fullfile(rootdir(n).folder, rootdir(n).name);
                trainorvaliddir = dir(trainorvalid);
                total = length(dir(fullfile(trainorvalid, "**/*.png")));
                for i = 1 : length(trainorvaliddir)
                     if trainorvaliddir(i).name ~= "."
                        if trainorvaliddir(i).name ~= ".."
                            if contains(trainorvaliddir(i).name, "hit") | contains(trainorvaliddir(i).name, "miss")
                                hitormiss = fullfile(trainorvaliddir(i).folder, trainorvaliddir(i).name);
                                hitormissddir = dir(hitormiss);
                                total = length(dir(fullfile(hitormiss, "**/*.png")));
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
                                                        step = step + 1;
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
                                              step = step + 100;
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
            datatocsv = [datatocsv;[center, radius]];
        end
    end
    csvsavepath = strrep(image,'99.png','trajectory.csv');
    csvsavepath = strrep(csvsavepath,readpath,savepath);
    [filepath,name,ext] = fileparts(csvsavepath);
    if ~exist(filepath, 'dir')
        mkdir(filepath);
    end
    csvwrite(csvsavepath,datatocsv);
end

function [centersBright,radiiBright] = getcirclecoordinates(path)
    if contains(path, "view1")
       radius = [8 15];
       sense = 0.99;
       edge = 0.1;
    else
       radius = [25 40];    
       sense = 0.98;
       edge = 0.2;
    end
    %it returns the mid point of the ball and radius
    rgb = imread(path);
    %gray_image = rgb2gray(rgb);
    %imshow(rgb)
    %13
    [centersBright,radiiBright,metricBright] = imfindcircles(rgb,radius, ...
        'ObjectPolarity','bright','Sensitivity',sense,'EdgeThreshold',edge);
    
    if contains(path, "view1")
        %don't need circle in the subject body if it is already thrown. 
        % it is probably error
        for i = length(radiiBright):-1:1
            if centersBright(i,1) >= 720 & centersBright(i,2) >= 270
                radiiBright(i) = [];
                centersBright(i,:) = [];
            end
        end
    else
       for i = length(radiiBright):-1:1
            if centersBright(i,2) >= 540
                radiiBright(i) = [];
                centersBright(i,:) = [];
            end
        end
    end
%         
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