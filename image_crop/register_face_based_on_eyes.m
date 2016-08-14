% Extracts the face region from an image based on provided eye coordinates
% 
% PROTOTYPE
% Y = register_face_based_on_eyes(X,eyes,size1)
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       eyes.x(1)=160;  %These coordinates were obtained using getpts(),
%       eyes.y(1)=184;  %but could also be read from a file
%       eyes.x(2)=193;
%       eyes.y(2)=177;
%       X=imread('sample_image.bmp');
%       Y = register_face_based_on_eyes(X,eyes,[128 100]);
%       figure,imshow(X,[]);
%       figure,imshow(uint8(Y),[]);
% 
% INPUTS:
% X                     - a grey-scale or color image of arbitrary size
% eyes				    - a structure containing the eye coordinates; the
%                         structure takes the following form:                           
%             eyes.x(1) - x coordinate of the images left eye 
%                         (this corresponds to the right eye in real life)
%             eyes.x(2) - x coordinate of the images right eye 
%                         (this corresponds to the left eye in real life)
%             eyes.y(1) - y coordinate of the images left eye 
%                         (this corresponds to the right eye in real life)
%             eyes.y(2) - y coordinate of the images right eye 
%                         (this corresponds to the left eye in real life)
% size1                 - a parameter determining the size of the output 
%                         image; it can take a single value, e.g., size=128, 
%                         or it can be a 1x2 matrix, e.g., size=[100 60].
%                         default: size1=[128 128]
%
% OUTPUTS:
% Y                     - a grey-scale or color facial image (depending on 
%                         the input format) extracted from the input image X 
%
% NOTES / COMMENTS
function Y = register_face_based_on_eyes(X,eyes,chin,size1)

%% Init
Y=[];



[a,b] = size(eyes.y);
if a==1 && b==2 || a==2 && b==1
else
    disp('The y coordinates were not defined properly. Please type "help register_face_based_on_eyes"')
    return;
end

%check size parameter
[a,b]=size(size1);
if a==1 && b==1
    size1 = [size1 size1];
elseif a==1 && b==2
    size1=size1;
else
    disp('The parameter defining the size of the output is not in the right format.')
    return;
end
    

%% Parse inputs

%convert to double
X=double(X);

[a,b,dummy] = size(X);
x_cent = ceil(b/2);
y_cent = ceil(a/2);

%these are the remains of my previous implementation
x_leva = double(eyes.x(1));
y_leva = double(eyes.y(1));
x_desna = double(eyes.x(2));
y_desna = double(eyes.y(2));
x_chin = double(chin.x(1));
y_chin = double(chin.y(1));

kk = (abs(y_leva-y_chin)+abs(y_desna-y_chin))/2;
%% Correct for rotation

%get rotation angle
kateta1 = abs(x_leva-x_desna);
kateta2 = abs(y_leva-y_desna);

if y_leva>y_desna
    angle = -(atan(kateta2/kateta1))*(180/pi);
elseif y_leva<y_desna
    angle = (atan(kateta2/kateta1))*(180/pi);
elseif y_leva==y_desna
    angle = 0;
end


%recompute the coordinates in the rotated image
if angle ==0
    Y=X;
    [c,d,e1]=size(Y);    
else
    Y=uint8(imrotate((X),angle,'bilinear'));
    [c,d,e]=size(Y);
    x_cent_rot = ceil(d/2);
    y_cent_rot = ceil(c/2);
    
    x_mid = round((x_leva+x_desna)/2);
    y_mid = round((y_leva+y_desna)/2);
    
    my_hypo = sqrt((x_leva-x_desna)^2+(y_leva-y_desna)^2);
    
    if x_mid == x_cent
     x_mid=x_mid+1;
    end

    if y_mid == y_cent
         y_mid=y_mid+1;
    end
    
    hipoten = sqrt((x_mid-x_cent)^2+(y_mid-y_cent)^2);
    x_koor = abs(x_mid-x_cent);
    if (x_mid > x_cent) & (y_mid<y_cent)        
        kot = acos(x_koor/hipoten)+(angle*pi/180);
        x1(1) = (x_cent_rot+cos(kot)*hipoten);
        y1(1) = (y_cent_rot-sin(kot)*hipoten);
    end   
    if (x_mid < x_cent) & (y_mid<y_cent)        
        kot = acos(x_koor/hipoten)-(angle*pi/180);
        x1(1) = (x_cent_rot-cos(kot)*hipoten);
        y1(1) = (y_cent_rot-sin(kot)*hipoten);
    end
    if (x_mid > x_cent) & (y_mid>y_cent)        
        kot = acos(x_koor/hipoten)-(angle*pi/180);
        x1(1) = (x_cent_rot+cos(kot)*hipoten);
        y1(1) = (y_cent_rot+sin(kot)*hipoten);
    end  
    if (x_mid < x_cent) & (y_mid>y_cent)        
        kot = acos(x_koor/hipoten)+(angle*pi/180);
        x1(1) = (x_cent_rot-cos(kot)*hipoten);
        y1(1) = (y_cent_rot+sin(kot)*hipoten);
    end
    
    x_leva = double(round(x1-(my_hypo/2))-1);
    x_desna = double(round(x1+(my_hypo/2))-1);
    y_leva = double(round(y1)-1);
    y_desna = double(round(y1)-1);
end


%% Extract facial region

% origin point for face frame
T_x = round((x_leva+x_desna)/2);
T_y = round((y_leva+y_desna)/2);
    
% frame coordinates calculation
d_ex = abs(x_desna-x_leva); 

% these determine the extracted region - in percentage (% = k*100) of the 
% interoccluar distance (You can play with these and get different facial areas)
%k1 = 0.6;       %these are the final coefficient, there will be no more change - the xm2vts database was extracted a bit differently (just a bit)
%k2 = 1.75;
% k3 = 0.9;
% k4 = 0.9;
k1 = 1.5;
k2 = 2;
k3 = 1.25;
k4 = 1.25;

% compute bounding box coordinates
% y_u = T_y-round(k1*d_ex);
% if y_u<=1
%     y_u = 1;
% end
% 
% y_l = T_y+round(k2*d_ex);
% if y_l>a
%     y_l=a;
% end
% 
% x_f = T_x-round(k3*d_ex);
% if x_f<1
%     x_f=1;
% end
% 
% x_s = T_x+round(k4*d_ex);
% if x_s>b
%     x_s=b;
% end

%y_u = T_y-round(kk*0.2);
%y_l = T_y+round(kk*0.9);

y_u = T_y-round(kk*0.5);
y_l = T_y+round(kk*1.1);
% y_l = T_y+round(kk*1.5);
%y_u = T_y-round(kk*0.3);
%y_l = min(T_y+round(kk*1.3), size(Y,1));
x_f = T_x-round(d_ex);
x_s = T_x+round(d_ex);
%x_f = T_x-round(kk*0.75);
%x_s = T_x+round(kk*0.75);

%disp([y_u, y_l, x_f, x_s]);
%extract final facial region
Y=Y(y_u:y_l, x_f:x_s,:);
Y=imresize(Y,size1,'bilinear');
Y=uint8(Y);