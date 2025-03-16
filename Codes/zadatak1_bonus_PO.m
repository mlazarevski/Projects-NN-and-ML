%% Ucitavanje
clear; close all; clc

rockPath = 'signs/rock';
paperPath = 'signs/paper';
scissorsPath = 'signs/scissors';

rockFiles = dir(fullfile(rockPath, '*.png'));
paperFiles = dir(fullfile(paperPath, '*.png'));
scissorsFiles = dir(fullfile(scissorsPath, '*.png'));

Nr = numel(rockFiles); Ns = numel(scissorsFiles); Np = numel(paperFiles);
rockImages = cell(1, Nr); paperImages = cell(1, Np); scissorsImages = cell(1, Ns);
rockLabels = zeros(1, Nr); paperLabels = ones(1, Np); scissorsLabels = 2 * ones(1, Ns);

for i = 1:Nr
    imagePath = fullfile(rockPath, rockFiles(i).name);
    rockImages{i} = rgb2hsv(imread(imagePath));
end

for i = 1:Np
    imagePath = fullfile(paperPath, paperFiles(i).name);
    paperImages{i} = rgb2hsv(imread(imagePath));
end

for i = 1:Ns
    imagePath = fullfile(scissorsPath, scissorsFiles(i).name);
    scissorsImages{i} = rgb2hsv(imread(imagePath));
end

%% Dobijanje obeležja
paperFeaturesRaw = processImages(paperImages);
rockFeaturesRaw = processImages(rockImages);
scissorFeaturesRaw = processImages(scissorsImages);

% save('KugliceObelezja.mat', 'paperFeaturesRaw', 'rockFeaturesRaw', 'scissorFeaturesRaw', 'Np', 'Nr', 'Ns');
%% Standardizacija
% load('KugliceObelezja.mat');

paperFeaturesRaw = paperFeaturesRaw(randperm(Np), :);
scissorFeaturesRaw = scissorFeaturesRaw(randperm(Ns), :);
rockFeaturesRaw = rockFeaturesRaw(randperm(Nr), :);

paperTrain = paperFeaturesRaw(1:round(0.7*end),:);
paperTest  = paperFeaturesRaw(round(0.7*end)+1:end,:);

scissorTrain = scissorFeaturesRaw(1:round(0.7*end),:);
scissorTest  = scissorFeaturesRaw(round(0.7*end)+1:end,:);

rockTrain = rockFeaturesRaw(1:round(0.7*end),:);
rockTest  = rockFeaturesRaw(round(0.7*end)+1:end,:);

%% Redukcija dimenzija na bazi matrice rasejanja

Mp = mean(paperTrain);
Ms = mean(scissorTrain);
Mr = mean(rockTrain);

Sp = cov(paperTrain);
Ss = cov(scissorTrain);
Sr = cov(rockTrain);

Pp = Np/(Np+Ns+Nr);
Ps = Ns/(Np+Ns+Nr);
Pr = Nr/(Np+Ns+Nr);

Sw = Pp*Sp+Ps*Ss+Pr*Sr;
M0 = Pp*Mp+Ps*Ms+Pr*Mr;
Sb = Pp*(Mp-M0)*(Mp-M0)'+Ps*(Ms-M0)*(Ms-M0)'+Pr*(Mr-M0)*(Mr-M0)';
S = (inv(Sw+Sb)*Sb);
[V, D] = eig(S);

figure;
stem(diag(D), 'filled');
grid on; grid minor
title('Sopstvene vrednosti');
xlabel('Indeks'); ylabel('Intenzitet [r.j.]');

A = V(:,5:end);
paperTrainSmall = A'*paperTrain';
scissorTrainSmall = A'*scissorTrain';
rockTrainSmall = A'*rockTrain';

paperTestSmall = A'*paperTest';
scissorTestSmall = A'*scissorTest';
rockTestSmall = A'*rockTest';

%% Test više hipoteza minimalne verovatnoće greške

Mp = mean(paperTrainSmall, 2);
Ms = mean(scissorTrainSmall, 2);
Mr = mean(rockTrainSmall, 2);

Sp = cov(paperTrainSmall');
Ss = cov(scissorTrainSmall');
Sr = cov(rockTrainSmall');

labels = [ones(1, length(paperTestSmall)), 2 * ones(1, length(scissorTestSmall)), 3 * ones(1, length(rockTestSmall))];
pred = [];
for x = [paperTestSmall scissorTestSmall rockTestSmall]
    qp = log(Pp) - 0.5*log(det(Sp)) - 0.5*(x-Mp)'*inv(Sp)*(x-Mp);
    qr = log(Pr) - 0.5*log(det(Sr)) - 0.5*(x-Mr)'*inv(Sr)*(x-Mr);
    qs = log(Ps) - 0.5*log(det(Ss)) - 0.5*(x-Ms)'*inv(Ss)*(x-Ms);
    [~, pred(end+1)] = max([qp, qs, qr]);
end

accuracy = sum(pred == labels) / length(pred);
disp(['Tačnost: ', num2str(accuracy)]);
% CM(I,J) known group labels I, predicted J.
C = confusionmat(labels, pred); 
disp('Matrica konfuzije');
classNames = {'Papir', 'Makaze', 'Kamen'};
confusionTable = array2table(C, 'RowNames', classNames, 'VariableNames', classNames);
display(confusionTable);

%% Makaze i kamen su najseparabilnije

Ms = mean(scissorFeaturesRaw);
Mr = mean(rockFeaturesRaw);

Ss = cov(scissorTrain);
Sr = cov(rockTrain);

P1 = Ns/(Ns+Nr);
P2 = Nr/(Ns+Nr);

Sw = P1*Ss+P1*Sr;
M0 = P1*Ms+P2*Mr;
Sb = P1*(Ms-M0)*(Ms-M0)'+P2*(Mr-M0)*(Mr-M0)';
S = (inv(Sw+Sb)*Sb);
[V, D] = eig(S);

figure;
stem(diag(D), 'filled');
grid on; grid minor
title('Sopstvene vrednosti');
xlabel('Indeks'); ylabel('Intenzitet [r.j.]');

A = V(:, 11:12);

s = A'*scissorFeaturesRaw';
r = A'*rockFeaturesRaw';

figure;
subplot(211);
hold all
histogram(r(1, :), 'FaceColor', 'r');
histogram(s(1, :), 'FaceColor', 'b');
title('Prvo obeležje');

subplot(212);
hold all
histogram(r(2, :), 'FaceColor', 'r');
histogram(s(2, :), 'FaceColor', 'b');
title('Drugo obeležje');

U = []; G = [ones(length(s)+length(r), 1)];
for x = r
    U(:,end+1) = -[1 x(1)^2 x(1)*x(2) x(2)^2]';
end
for x = s
    U(:,end+1) = [1 x(1)^2 x(1)*x(2) x(2)^2]';
end

W = pinv(U)'*G;

x1_range = linspace(min([r(1, :), s(1, :)]), max([r(1, :), s(1, :)]), 100);
x2_range = linspace(min([r(2, :), s(2, :)]), max([r(2, :), s(2, :)]), 100);
[x1, x2] = meshgrid(x1_range, x2_range);
hx = W(1) + W(2) * x1.^2 + W(3) * x1 .* x2 + W(4) * x2.^2;

figure;
title('Kvadratni klasifikator metodom željenog izlaza')
hold all
grid on
scatter(r(1,:),r(2,:), 'ro');
scatter(s(1,:),s(2,:),'bv');
contour(x1, x2, hx, [0, 0], 'LineColor', 'g');
legend('kamen','makaze')

%% Prikaz koraka u obradi slike
depictProcess('Kamen', rockImages{randi([1, Nr])})
depictProcess('Papir', paperImages{randi([1, Np])})
depictProcess('Makaze', scissorsImages{randi([1, Ns])})

%% Funkcija za prikaz obrade slike
function depictProcess(class, img)
    h = fspecial('gaussian', [5, 5], 1.0);

    figure;
    sgtitle(class)    
    
    subplot(221);
    imshow(img);
    title('H komponenta HSV');
    
    temp = imfilter(img, h, 'conv');
    subplot(222);
    imshow(temp);
    title('Gausov filter');
    
    temp = 1-imbinarize(temp(:,:,1));
    subplot(223);
    imshow(temp);
    title('Binarizacija Otsu metodom');
    
    temp = imerode(imdilate(temp, strel('disk', 5)), strel('disk', 5));
    subplot(224);
    imshow(temp);
    title(['Morfološko uklanjanje šuma, površina: ' num2str(sum(sum(temp)))])
    
    stats = regionprops(temp, 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Orientation');
    majorAxis = stats.MajorAxisLength;
    minorAxis = stats.MinorAxisLength;
    orientation = stats.Orientation;
    centroid = stats.Centroid;
    
    hold on;
    plot(centroid(1), centroid(2), 'ro');  % plot centroid
    plot([centroid(1), centroid(1) + 0.5 * majorAxis * cosd(orientation)], ...
         [centroid(2), centroid(2) + 0.5 * majorAxis * sind(orientation)], 'r-', 'LineWidth', 2);
    
    plot([centroid(1), centroid(1) + 0.5 * minorAxis * cosd(orientation + 90)], ...
             [centroid(2), centroid(2) + 0.5 * minorAxis * sind(orientation + 90)], 'g:', 'LineWidth', 2);
    
    hold off;

end

%% Funkcija za obradu slike
function L = processImages(batch)
    N = length(batch);
    L = zeros(N, 12);

    h = fspecial('gaussian', [5, 5], 1.0);
    
    for i=1:N
        temp = imfilter(batch{i}(:,:,1), h, 'conv');
        temp = imbinarize(temp);
        temp = imdilate(temp, strel('disk', 5));
        temp = imerode(temp, strel('disk', 5));

        stats = regionprops(1-temp, 'Centroid', 'MajorAxisLength','MinorAxisLength','Orientation','Eccentricity','FilledArea','Area','ConvexArea','Circularity','Solidity','Extent');
        
        L(i, :) = [stats.Centroid, stats.MajorAxisLength, stats.MinorAxisLength, stats.Orientation, stats.Eccentricity, stats.FilledArea, stats.Area, stats.ConvexArea, stats.Circularity, stats.Solidity, stats.Extent];
    end
end




