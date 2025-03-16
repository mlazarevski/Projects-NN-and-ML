clear; close all; clc

emptyPath = 'Kuglice/Prazne';
fullPath = 'Kuglice/Pune';

emptyFiles = dir(fullfile(emptyPath, '*.bmp'));
fullFiles = dir(fullfile(fullPath, '*'));

Ncolors = numel(fullFiles);
Nf = 1052; 
Ne = numel(emptyFiles);
n = 600; 

fullImages = cell(1, Nf);
emptyImages = cell(1, Ne);

ColorLabels = zeros(1, Nf);
cnt = 0; 

for i = 1:Ncolors
    if ~startsWith(fullFiles(i).name, '.') && fullFiles(i).isdir
        bojaPath = fullfile(fullPath, fullFiles(i).name);
        files = dir(fullfile(bojaPath, '*.bmp'));
        for j = 1:numel(files)
            imagePath = fullfile(bojaPath, files(j).name);
            cnt = cnt + 1; ColorLabels(cnt) = i;
            fullImages{cnt} = rgb2hsv(imread(imagePath));
        end
    end
end

ColorLabels = ColorLabels + 1 - min(ColorLabels);

for i = 1:Ne
    imagePath = fullfile(emptyPath, emptyFiles(i).name);
    emptyImages{i} = rgb2hsv(imread(imagePath));
end

%% Prikaz punih i praznih obeležja
% Gornja leva četvrtina slike je karakteristična po svojoj varijansi
% Puna
figure;
sgtitle('H S V')
Nprikaz = 3;
Mprikaz = 5;
for k = 1:3:Nprikaz*Mprikaz
    idx = randi([1, Nf]); temp = fullImages{idx}(1:n/2,1:n/2,:);
    subplot(Mprikaz,Nprikaz,k);
    imshow(temp(:,:,1));
    title(num2str(var(reshape(temp(:,:,1), [], 1))))
    subplot(Mprikaz,Nprikaz,k+1);
    imshow(temp(:,:,2));
    title(num2str(var(reshape(temp(:,:,2), [], 1))))
    subplot(Mprikaz,Nprikaz,k+2);
    imshow(temp(:,:,3));
    title(num2str(var(reshape(temp(:,:,3), [], 1))))
end

% Prazna
figure;
sgtitle('H S V')
Nprikaz = 3;
Mprikaz = 5;
for k = 1:3:Nprikaz*Mprikaz
    idx = randi([1, Ne]); temp = emptyImages{idx}(1:n/2,1:n/2,:);
    subplot(Mprikaz,Nprikaz,k);
    imshow(temp(:,:,1));
    title(num2str(var(reshape(temp(:,:,1), [], 1))))
    subplot(Mprikaz,Nprikaz,k+1);
    imshow(temp(:,:,2));
    title(num2str(var(reshape(temp(:,:,2), [], 1))))
    subplot(Mprikaz,Nprikaz,k+2);
    imshow(temp(:,:,3));
    title(num2str(var(reshape(temp(:,:,3), [], 1))))
end

% Pakovanje obeležja
feature1Full = zeros(Nf,3);
feature1Empty = zeros(Ne,3);
for i = 1:Nf
    temp = fullImages{i}(1:n/2, 1:n/2, :);    
    for k=1:3
        feature1Full(i,k) = var(reshape(temp(:,:,k), [], 1));
    end
end

for i = 1:Ne
    temp = emptyImages{i}(1:n/2, 1:n/2, :);    
    for k=1:3
        feature1Empty(i,k) = var(reshape(temp(:,:,k), [], 1));
    end
end

% Prikaz separabilnosti
figure
sgtitle('Histogrami varijansi pojedinačnih kanala HSV za kuglice')
for k=1:3
    subplot(3,1,k)
    grid on; grid minor;
    histogram(feature1Full(:,k), 'BinEdges', 0:0.0001:0.01, 'FaceColor', 'red', 'EdgeColor', 'none');
end

figure
sgtitle('Histogrami varijansi pojedinačnih kanala HSV za prazne slike')
for k=1:3
    subplot(3,1,k)
    grid on; grid minor;
    histogram(feature1Empty(:,k), 'BinEdges', 0:0.0001:0.01, 'FaceColor', 'red', 'EdgeColor', 'none');
end

% Nalaženje trivijalnog linearnog klasifikatora
X = [feature1Full(:,1); feature1Empty(:,1)];
L = [ones(size(feature1Full, 1), 1); zeros(size(feature1Empty, 1), 1)];

% Zbog lošeg labeliranja ne možemo koristiti SVM 
T = max(feature1Empty(:,1)); %T = 8.387178323937491e-04

Lf = feature1Full(:,1) > T;
Le = feature1Empty(:,1) <= T;

% Demonstracija rada algoritma
i = find(Lf == 0);

figure;
imshow(fullImages{i(2)});
title('Dobro klasifikovan, a pogrešno labeliran');

figure;
imshow(fullImages{i(1)});
title('Dobro labeliran, a pogrešno klasifikovan');

%
figure;
Nprikaz = 3;
Mprikaz = 3;
for k = 1:3:Nprikaz*Mprikaz
    idx = randi([1, Nf]); temp = fullImages{idx};
    subplot(Mprikaz,Nprikaz,k);
    imshow(temp(:,:,1));
    subplot(Mprikaz,Nprikaz,k+1);
    imshow(temp(:,:,2));
    subplot(Mprikaz,Nprikaz,k+2);
    imshow(temp(:,:,3));
end

%% Detekcija boje i maksimalne separabilnosti kuglice i pozadine

ColorFeaturesBalls = zeros(length(ColorLabels), 3);
ColorFeaturesBackground = zeros(length(ColorLabels), 3);
for k = 1:Nf
    for l = 1:3
        ColorFeaturesBackground(k,l) = median(median(fullImages{k}(1:n/4,1:n/4,l)));
        ColorFeaturesBalls(k,l) = median(median(fullImages{k}(n/4:3*n/4,n/4:3*n/4,l)));
    end
end

separability = ColorFeaturesBackground - ColorFeaturesBalls;
channels = zeros(1, length(ColorLabels)); 
for k = unique(ColorLabels)
    idx = (k == ColorLabels);
    curSeparability = abs(mean(separability(idx, :)));
    [~, curChannel] = max(curSeparability);
    channels(idx) = curChannel;
end

figure; 
for l = 1:3
    subplot(4,1,l)
    hold all
    plot(ColorFeaturesBackground(:,l)); plot(ColorFeaturesBalls(:, l))
end
subplot(414)
plot(channels)


%%
meanBallColor = zeros(1, length(unique(ColorLabels)));
meanBackgroundColor = zeros(1, length(unique(ColorLabels)));
for k = unique(ColorLabels)
    k
    idx = (k == ColorLabels);
    meanBallColor(k)       = mean(ColorFeaturesBalls(idx, mode(channels(idx))));
    meanBackgroundColor(k) = mean(ColorFeaturesBackground(idx, mode(channels(idx))));
end
SVMthreshold = (meanBallColor + meanBackgroundColor)/2;

%% 1, 2, 3,4, 6,7, 9, krnja 8, malo manje 5
load('Zadatak4Klasifikacija.mat');
b = [2 3 4 5 6 7 9];
Imax = find(ismember(ColorLabels, [1 4 9]), 1, 'last');
Imin = find(ismember(ColorLabels, [1 4 9]), 1, 'first');
filterSize = 10; 
close all

Nprikaz = 2;
Mprikaz = 3;
WrongPrediction = find(Predicted ~= X);
for k = 1:3:Nprikaz*Mprikaz
    for l = 1:3
        %zero_indices = find(X == 0);
        %idx = zero_indices(randi(numel(zero_indices)));

        idx = WrongPrediction(randi(length(WrongPrediction)));        
        
        color = ColorLabels(idx);
        tempHSV = fullImages{idx};   tempHSVgap = tempHSV;
        temp = tempHSV(:,:,channels(idx));    

        figure(1)
        title(num2str(X(idx)))
        subplot(Mprikaz,Nprikaz,k+l-1);
        imshow(hsv2rgb(tempHSV));

        m = median(median(temp(n/4:3*n/4,n/4:3*n/4)));
        if meanBallColor(color) < meanBackgroundColor(color)
            temp = 1 - temp;
        end
        temp = temp.*(temp > SVMthreshold(color));
        tempHSV = tempHSV.*(temp > SVMthreshold(color));

        figure(2)
        subplot(Mprikaz,Nprikaz,k+l-1);
        imshow(temp, []);

        connected_components = bwlabel(temp > 0.01);
        props = regionprops(connected_components, 'Area');
        [~, i] = max([props.Area]);
        temp = temp.*(connected_components == i);
        tempHSV = tempHSV.*(connected_components == i);


        figure(3)
        subplot(Mprikaz,Nprikaz,k+l-1);
        imshow(temp,  []);
        
        temp = medfilt2(temp, [filterSize, filterSize]);

        gradientMagnitude = imgradient(temp, 'sobel');
        columnSums = sum(gradientMagnitude, 1);
        columnSumsGrad = diff([columnSums(1) columnSums]).^2;
        rowSums = sum(gradientMagnitude, 2);
        rowSumsGrad = diff([rowSums(1); rowSums]).^2;

        figure(4)
        sgtitle('columnSumsGrad')
        subplot(Mprikaz,Nprikaz,k+l-1);
        plot(columnSumsGrad)

        figure(5)
        sgtitle('rowSumsGrad')
        subplot(Mprikaz,Nprikaz,k+l-1);
        plot(rowSumsGrad)

 
        
        Left = 50; Right = 50;
        for m = 50:300
            if columnSumsGrad(m) > 0
                Left = m;
                break;
            end
        end
        for m = n:-1:300
            if columnSumsGrad(m) > 0
                Right = m;
                break;
            end
        end

        Up = 100; Down = 100;
        for m = 100:300
            if rowSumsGrad(m) > 0
                Up = m;
                break;
            end
        end
        for m = n:-1:300
            if rowSumsGrad(m) > 0
                Down = m;
                break;
            end
        end


        temp = temp(Up:Down,Left:Right);
        tempHSV = tempHSV(Up:Down,Left:Right,:);

        temp = medfilt2(temp, [filterSize, filterSize]);
    
        figure(6)
        subplot(Mprikaz,Nprikaz,k+l-1);
        imshow(imgradient(imbinarize(temp)), []);


        

        roi = temp(1:30,:)>0.001;
        width  = 1 : size(temp,2);
        rowSums = mean(roi(15:end, :));
        rw = round(sum(width .* rowSums) / sum(rowSums));


        roi = temp(:, 1:30)>0.001;
        heigth  = 1 : size(temp,1);
        columnsSums = mean(roi(:, 15:end));
        rh = round(sum(width .* rowSums) / sum(rowSums));

        r = max(rw,rh);

        maxSize = min(size(temp));
        crop = min(2*r,maxSize);
        temp = temp(1:crop,1:crop);

        figure(7)
        subplot(Mprikaz,Nprikaz,k+l-1);
        imshow(temp);

        perimeter = bwperim(imbinarize(temp));
        temp = imfill(perimeter, 'holes');
        
        figure(8)
        subplot(Mprikaz,Nprikaz,k+l-1);
        imshow(temp);

        roi = imgradient(temp) .* [ones(floor(crop/3), crop); [ones(ceil(crop*2/3), floor(crop/3)) zeros(ceil(crop*2/3), ceil(crop*2/3))]];
        [rows, cols] = find(roi == 1);
        points = [cols, rows]; 
    
        x0 = crop/2; y0 = crop/2; r0 = crop/2;
        residuals = @(params) sum((sqrt((points(:,1)-params(1)).^2 + (points(:,2)-params(2)).^2) - params(3)).^2);
        options = optimoptions('lsqnonlin', 'Display', 'off');
        params = lsqnonlin(residuals, [x0, y0, r0], [], [], options);

        xCenter = params(1);
        yCenter = params(2);
        radius = params(3);
            
        figure(8)
        hold on;
        th = linspace(0, 2*pi, 100);
        x = xCenter + radius * cos(th);
        y = yCenter + radius * sin(th);
        plot(x, y, '-'); % Plot fitted circle
        hold off;


        r = radius;



        [x, y] = meshgrid(1:crop, 1:crop);
        distances = sqrt((x - xCenter).^2 + (y - yCenter).^2);
        distances_inside =  (distances <= r) & (temp);
        distances_outside = (distances > r) & (temp);





    end

end

%%
L = zeros(19, length(ColorLabels)); cnt_L = 1;
filterSize = 10; 

for color = unique(ColorLabels)
    for idx = find(ColorLabels == color)

        temp = fullImages{idx}(:,:,channels(idx));    

        m = median(median(temp(n/4:3*n/4,n/4:3*n/4)));
        if meanBallColor(color) < meanBackgroundColor(color)
            temp = 1 - temp;
        end

        temp = temp.*(temp > SVMthreshold(color));

        connected_components = bwlabel(temp > 0.01);
        props = regionprops(connected_components, 'Area');
        [~, i] = max([props.Area]);
        temp = temp.*(connected_components == i);
        
        temp = medfilt2(temp, [filterSize, filterSize]);

        gradientMagnitude = imgradient(temp, 'sobel');
        columnSums = sum(gradientMagnitude, 1);
        columnSumsGrad = diff([columnSums(1) columnSums]).^2;
        rowSums = sum(gradientMagnitude, 2);
        rowSumsGrad = diff([rowSums(1); rowSums]).^2;

         Left = 50; Right = 50;
        for m = 50:300
            if columnSumsGrad(m) > 0
                Left = m;
                break;
            end
        end
        for m = n:-1:300
            if columnSumsGrad(m) > 0
                Right = m;
                break;
            end
        end

        Up = 100; Down = 100;
        for m = 100:300
            if rowSumsGrad(m) > 0
                Up = m;
                break;
            end
        end
        for m = n:-1:300
            if rowSumsGrad(m) > 0
                Down = m;
                break;
            end
        end


        temp = temp(Up:Down,Left:Right);

        temp = medfilt2(temp, [filterSize, filterSize]);
    
        roi = temp(1:30,:)>0.001;
        width  = 1 : size(temp,2);
        rowSums = mean(roi(15:end, :));
        rw = round(sum(width .* rowSums) / sum(rowSums));


        roi = temp(:, 1:30)>0.001;
        heigth  = 1 : size(temp,1);
        columnsSums = mean(roi(:, 15:end));
        rh = round(sum(width .* rowSums) / sum(rowSums));

        r = max(rw,rh);

        maxSize = min(size(temp));
        crop = min(2*r,maxSize);
        temp = temp(1:crop,1:crop);
        
        temp = imbinarize(temp);
        perimeter = bwperim(temp);
        temp = imfill(perimeter, 'holes');

        roi = imgradient(temp) .* [ones(floor(crop/3), crop); [ones(ceil(crop*2/3), floor(crop/3)) zeros(ceil(crop*2/3), ceil(crop*2/3))]];
        [rows, cols] = find(roi == 1);
        points = [cols, rows]; 
    
        x0 = crop/2; y0 = crop/2; r0 = crop/2;
        residuals = @(params) sum((sqrt((points(:,1)-params(1)).^2 + (points(:,2)-params(2)).^2) - params(3)).^2);
        options = optimoptions('lsqnonlin', 'Display', 'off');
        params = lsqnonlin(residuals, [x0, y0, r0], [], [], options);

        xCenter = params(1);
        yCenter = params(2);
        radius = params(3);


        r = radius;


        [x, y] = meshgrid(1:crop, 1:crop);
        distances = sqrt((x - xCenter).^2 + (y - yCenter).^2);
        distances_inside =  (distances <= r) & (temp);
        distances_outside = (distances > r) & (temp);

                
        sums_inside = zeros(8, 1);
        sums_outside = zeros(8, 1);
       
        for i = 1:crop
            for j = 1:crop
                angle = atan2d(xCenter - i, j - yCenter);  
                
                if angle < 0
                    angle = angle + 360;
                end
                
                sector = floor(angle / 45) + 1;

                    sums_inside(sector) = sums_inside(sector) + distances_inside(i, j);
                    sums_outside(sector) = sums_outside(sector) + distances_outside(i, j);
            end
        end

             L(:, cnt_L) = [[sums_inside; sums_outside]./(crop^2); params'];
             cnt_L = cnt_L + 1;
             idx

    end
end


% save('Zadatak4Obelezja.mat', 'L');
% load('Zadatak4Klasifikacija.mat'); rucno radjeno ne brisati!!!
% load('Zadatak4Obelezja.mat');

%%
close all
Predicted = [];
for color = unique(ColorLabels)
    ColorsUsed = ColorLabels == color;
    Y = X(ColorsUsed);
    F = zscore(L(:,ColorsUsed));
    
    idx_X1 = find(Y == 1);
    idx_X0 = find(Y == 0);
    mu_X1 = mean(F(:, idx_X1), 2);
    covariance_X1 = cov(F(:, idx_X1)');
    
    distances_X0 = zeros(1, length(idx_X0));
    distances_X1 = zeros(1, length(idx_X1));
    
    for i = 1:length(idx_X0)
        distances_X0(i) = sqrt((F(:, idx_X0(i)) - mu_X1)' * pinv(covariance_X1) * (F(:, idx_X0(i)) - mu_X1));
    end
    
    for i = 1:length(idx_X1)
        distances_X1(i) = sqrt((F(:, idx_X1(i)) - mu_X1)' * pinv(covariance_X1) * (F(:, idx_X1(i)) - mu_X1));
    end
    
    distances = [distances_X0, distances_X1];
    labels = [zeros(1, length(distances_X0)), ones(1, length(distances_X1))];
    
    bestF1 = 0;
    bestThreshold = 0;
    
    for i = 1:length(distances)
        threshold = distances(i);
        
        predictions = zeros(size(distances));
        predictions(distances <= threshold) = 1;
        
        TP = sum(predictions(labels == 1));
        FP = sum(predictions(labels == 0));
        FN = sum(labels) - TP;
        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
    
        F1 = 2 * precision * recall / (precision + recall);
    
        if F1 > bestF1
            bestF1 = F1;
            bestThreshold = threshold;
        end
    end
    
    predictions = zeros(size(distances));
    predictions(distances <= bestThreshold) = 1;
    Predicted = [Predicted predictions];
    
    TP = sum(predictions(labels == 1));
    FP = sum(predictions(labels == 0));
    FN = sum(labels) - TP;
    TN = sum(~labels) - FP;
    
    P = TP / (TP + FN);
    S = TN / (TN + FP);
    accuracy= sum(predictions == labels)/length(labels);
    display([round(color) [P S accuracy].*100])
end



