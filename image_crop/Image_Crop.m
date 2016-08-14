clear
%% Image crop
% modified based on Rotate face paper's code
data = '/esat/visicsrodata/datasets/MultiPIE/Multi-Pie/Multi-Pie/data'; %data path 
posdata = '/esat/ruchba/xjia/image_generation/multi_pie/MultiPIE_annotations/Multipie-Annotations'; %parameter path
img_size = [60,60] % [64,64], [60,60]

jhseq = zeros(346,1);
for iiii = 1 %session number, in this experiment, we use only 1
    % for id = 1:lastID
    if iiii == 1
        lastId = 250
    elseif iiii==2
        lastId = 292
    elseif iiii==3
        lastId = 346
    elseif iiii==4
        lastId = 346
    end
    
    imgdir = [data, '/session0' num2str(iiii) '/multiview'];
    cat = dir(imgdir);
    indexm = zeros(lastId,1);
    for i = 3:length(cat)
        indexm(str2num(cat(i).name),1) = 1;
    end
    negindex = find(indexm==0);
    for id = 1:lastId
        
        if length(find(negindex==id))==1
            continue;
        end
        if jhseq(id,1)==1
            continue;
        end
        jhseq(id,1)=1;
        
        %     if(ID_idx(id)==0)
        %         continue;
        %     end
        ID = sprintf('%.3d', id);
        %     datasetNum = mod(ID_idx(id),10);
        cnt_pos_pair = 0;
        %     catIndarr = [1 2 3 4 5 7 8 9 10 11 12 14 15];
        catIndarr = [2 3 4 5 7 10 11 12 14];
        
        for catInd = catIndarr
            cnt = 0;
            %        while cnt< 2
            %             session = find(attendance(ID_idx(id),:));
            session=[iiii];
            if(length(session)==4)
                mat24 = [1 2 3 4; 1 1 1 1]; % [session4; recnum2]
            elseif(length(session)==3)
                if(isequal(session, [1 2 3]))
                    mat24 = [1 1 2 3; 1 2 2 3];
                elseif(isequal(session, [1 2 4]))
                    mat24 = [1 2 4 4; 2 2 1 2];
                elseif(isequal(session, [2 3 4]))
                    mat24 = [2 3 3 4; 2 2 3 1];
                elseif(isequal(session, [1 3 4]))
                    mat24 = [1 3 4 4; 2 3 1 2];
                end
            elseif(length(session)==2)
                if(isequal(session, [1 2]))
                    mat24 = [1 1 2 2; 1 2 1 2];
                elseif(isequal(session, [1 3]))
                    mat24 = [1 1 3 3; 1 2 1 3];
                elseif(isequal(session, [1 4]))
                    mat24 = [1 1 4 4; 1 2 1 2];
                elseif(isequal(session, [2 3]))
                    mat24 = [2 2 3 3; 1 2 2 3];
                elseif(isequal(session, [2 4]))
                    mat24 = [2 2 4 4; 1 2 1 2];
                elseif(isequal(session, [3 4]))
                    mat24 = [3 3 4 4; 1 2 3 2];
                end
            else
                s = session;
                if(session==1 || session==2 || session==4|| session==3)
                    %                     mat24 = [s s s s; 1 2 1 2];
                    mat24 = [s s; 1 2];
                else
                    mat24 = [s s s s; 1 2 3 1];
                end
            end
            tmp = {};
            %         for i = 1:length(mat24)
            for i = 1:1 %2 % only use recording01
                session4 = mat24(1,i);
                recnum2 = mat24(2,i); 
                
                ID_1 = ID;
                recnum_2 = num2str(recnum2);
                session_4 = num2str(session4);
                imgdir = [data '/session0' session_4 '/multiview/' ID_1 '/0' recnum_2 '/'];
                posdir = [posdata '/session0' session_4 '/multiview/' ID_1 '/0' recnum_2 '/'];
                cat = dir(imgdir);
                cat = cat(3:end);
                %        catInd = randperm(length(cat)-2,1) + 2;
                
                cam = cat(catInd).name
                imgdir2 = [imgdir cam];
                posdir2 = [posdir cam];
                img_pose = [posdir2 '/' ID_1 '_0' session_4 '_0' recnum_2 '_' cam([1:2,4]) '_00.pos']
                cat2 = dir(imgdir2);
                imgInd_index = randperm(length(cat2)-2)-1;
                for imgInd = imgInd_index
                    num = sprintf('%.2d', imgInd);
                    img = [imgdir2 '/' ID_1 '_0' session_4 '_0' recnum_2 '_' cam([1:2,4]) '_' num '.png'];
                    imgname = [ID_1 '_0' session_4 '_0' recnum_2 '_' cam([1:2,4]) '_' num '.PNG'];
                    if(isempty(find(ismember(tmp, imgname), 1)))
                        tmp{cnt+1} = imgname;
                        if(strcmp(cam,'01_0')||strcmp(cam,'09_0')||strcmp(cam,'11_0')||...
                                strcmp(cam,'12_0')||strcmp(cam,'20_0')||strcmp(cam,'24_0'))
                            %a = textread(img_pose);
                            fid = fopen(img_pose);
                            a = textscan(fid, '%d %d');
                            a = [a{1} a{2}];
                            fclose(fid);
                            eyes = struct('x', a(2,1), 'y', a(2,2));
                            chin = struct('x', a(7,1), 'y', a(7,2));
                            X=imread(img);
                            Y = register_face_based_on_eyes2(X,eyes,chin,[img_size(1) img_size(2)]);%crop size [32 32] for setting 2 and [60 60] for setting 1
                        elseif(strcmp(cam,'04_1')||strcmp(cam,'05_0')||strcmp(cam,'05_1')||...
                                strcmp(cam,'13_0')||strcmp(cam, '14_0'))
                            %a = textread(img_pose);
                            fid = fopen(img_pose);
                            a = textscan(fid, '%d %d');
                            a = [a{1} a{2}];
                            fclose(fid);
                            eyes = struct('x', a(2:3,1), 'y', a(2:3,2));
                            chin = struct('x', a(13,1), 'y', a(13,2));
                            X=imread(img);
                            Y = register_face_based_on_eyes(X,eyes,chin, [img_size(1) img_size(2)]);%crop size [32 32] for setting 2 and [60 60] for setting 1
                        elseif(strcmp(cam,'19_0')||strcmp(cam,'08_0'))
                            fid = fopen(img_pose);
                            a = textscan(fid, '%d %d');
                            a = [a{1} a{2}];
                            fclose(fid);
                            eyes = struct('x', a(2:3,1), 'y', a(2:3,2));
                            chin = struct('x', a(9,1), 'y', a(9,2));
                            X=imread(img);
                            Y = register_face_based_on_eyes(X,eyes,chin, [img_size(1) img_size(2)]);%crop size [32 32] for setting 2 and [60 60] for setting 1
                        end
                        
                        dataDir = ['/esat/ruchba/xjia/image_generation/multi_pie/face_aligned_60_color']; %_color, save path : newdataset3232_small for setting 2 and newdataset6060 for setting 1
                        img = [dataDir '/' ID_1 '_0' session_4 '_0' recnum_2 '_' cam([1:2,4]) '_' num '.PNG'];
                        %Y = rgb2gray(Y);
                        imwrite(Y, img, 'png');
                        cnt = cnt+1;
                        disp([ID_1, ': ', num2str(cnt_pos_pair) ' saved ' img]);
                        cnt_pos_pair = cnt_pos_pair+1;
                    end
                end
            end
        end
    end
end





