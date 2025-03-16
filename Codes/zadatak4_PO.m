%% Parametri i graficki prikaz
clear; close all; clc;

N = 500;
M1 = [-2 6]';
S1 = [2 0.5; 0.5 10];
M2 = [6 1]';
S2 = [1.5 0.1; 0.1 2];
M3 = [10 10]';
S3 = [2 0.9; 0.9 2];
M4 = [15 0]';
S4 = [4 0.2; 0.2 7];


K1 = mvnrnd(M1,S1,N)';
K2 = mvnrnd(M2,S2,N)';
K3 = mvnrnd(M3,S3,N)';
K4 = mvnrnd(M4,S4,N)';


my_depictClasses(K1,K2,K3,K4)
X = [K1 K2 K3 K4];

%% C-means
L = randi(4,1,length(X));
[Niter, L] = my_cMeans(X, L);

E1 = X(:,L == 1);
E2 = X(:,L == 2);
E3 = X(:,L == 3);
E4 = X(:,L == 4);

my_depictClasses(E1,E2,E3,E4)

%% Normal Decomposition
X = [K1 K2 K3 K4];
L = randi(4,1,length(X));
[Niter, L] = my_normalDecomposition(X, L);

E1 = X(:,L == 1);
E2 = X(:,L == 2);
E3 = X(:,L == 3);
E4 = X(:,L == 4);

my_depictClasses(E1,E2,E3,E4)

%% Branch and Bound
[L, Y] = my_BranchAndBound(X, L);

E1 = Y(:,L == 1);
E2 = Y(:,L == 2);
E3 = Y(:,L == 3);
E4 = Y(:,L == 4);

my_depictClasses(E1,E2,E3,E4)

%% Maximum Likelihood 
X = [K1 K2 K3 K4];
L = randi(4, 1, 4*N);
[Niter, L] = my_MaxLikelihoodClustering(X, L);

E1 = X(:,L == 1);
E2 = X(:,L == 2);
E3 = X(:,L == 3);
E4 = X(:,L == 4);

my_depictClasses(E1,E2,E3,E4)

%% Performanse za linearno separabilne klase
evaluate_clustering_performance(@my_cMeans, 'klasifikatora c-means', N, X)
evaluate_clustering_performance(@my_MaxLikelihoodClustering, 'klasifikatora maksimalne verodostojnosti', N, X)
evaluate_clustering_performance(@my_normalDecomposition, 'kvadratnog klasifikatora', N, X)
% Besmislica 
% evaluate_clustering_performance(@my_BranchAndBound, 'Branch and Bound klasifikatora', N, X)


%% Linearno neseparabilne klase: Parametri i graficki prikaz

N = 500;

angle = 2 * pi * rand(1, N);
r = 0 + 1 * sqrt(rand(1, N)); 
x1 = 0.5 * r .* cos(angle);
x2 = 1.1 * r .* sin(angle);

angle = 2 * pi * rand(1, N);
r = 5 + 5 * sqrt(rand(1, N)); 
y1 = 0.50 * r .* cos(angle);
y2 = 0.25 * r .* sin(angle);


figure;
plot(x1, x2, 'bv');
hold on;
plot(y1, y2, 'ro');
axis equal;
xlabel('X');
ylabel('Y');
title('Linearno neseparabilne klase');
legend('K1', 'K2');

Y = [x1 y1; x2 y2];

L = randi(2, 1, 2*N);
[Niter, L] = my_MaxLikelihoodClustering(Y, L);

E1 = Y(:,L == 1);
E2 = Y(:,L == 2);

figure
hold all
title('Klase')
scatter(E1(1,:),E1(2,:),'m*');
scatter(E2(1,:),E2(2,:),'bo');

grid on
grid minor
xlabel('x1');
ylabel('x2');
legend('K1','K2');


%% Nelinearna analiza
evaluate_clustering_performance2(@my_MaxLikelihoodClustering, 'klasifikatora maksimalne verodostojnosti', N, Y)
evaluate_clustering_performance2(@my_normalDecomposition, 'kvadratnog klasifikatora', N, Y)

%% Evaluator
function evaluate_clustering_performance(my_fnc, name, N, X)
    Ntest = 10;
    accuracies = zeros(1, Ntest); times = zeros(1, Ntest);
    Niter = zeros(1, Ntest);
    
    L_true = [ones(1, N) 2*ones(1, N) 3*ones(1, N) 4*ones(1, N)];
    
    for k = 1:Ntest
        Nclass = max(4, k);
        L = randi(Nclass, 1, 4*N);
        tic;
        [Niter(k), L] = my_fnc(X, L);
        time = toc; 
        
        modified_L = L;
        for i = 1:4
            chunk = L((i-1)*500 + 1 : i*500);
            median_label = mode(chunk);
            modified_L((i-1)*500 + find(chunk == median_label)) = i;
            modified_L((i-1)*500 + find(chunk ~= median_label)) = median_label;
        end
        L = modified_L;
        
        accuracies(k) = 100*sum(L_true == L) / (4 * N);
        times(k) = time;
    end
    
    figure;
    sgtitle(['Osetljivost na broj klasa ' name])
    
    subplot(311)
    plot(times)
    title('Vreme izvršavanja')
    ylabel('Vreme [s]')
    xlabel('Broj klasa')
    
    subplot(312)
    plot(Niter)
    title('Ukupan broj iteracija algoritma')
    ylabel('Broj iteracija algoritma')
    xlabel('Broj klasa')
    
    subplot(313)
    plot(accuracies)
    title('Tačnost izvršavanja')
    ylabel('Tačnost [%]')
    xlabel('Broj klasa')

    accuracies = zeros(1, Ntest); times = zeros(1, Ntest);
    Niter = zeros(1, Ntest);
    
    L_true = [ones(1, N) 2*ones(1, N) 3*ones(1, N) 4*ones(1, N)];
    
    for k = 1:Ntest
        L = randi(4, 1, 4*N);
        tic;
        [Niter(k), L] = my_MaxLikelihoodClustering(X, L);
        time = toc; 
        
        modified_L = L;
        for i = 1:4
            chunk = L((i-1)*500 + 1 : i*500);
            median_label = mode(chunk);
            modified_L((i-1)*500 + find(chunk == median_label)) = i;
            modified_L((i-1)*500 + find(chunk ~= median_label)) = median_label;
        end
        L = modified_L;
        
        accuracies(k) = 100*sum(L_true == L) / (4 * N);
        times(k) = time;
    end
    
    display(['Srednje vreme: ' num2str(mean(times)) 's']);
    display(['Srednji broj iteracija: ' num2str(mean(Niter))]);
    display(['Srednja tačnost: ' num2str(mean(accuracies)) '%']);
    
    figure;
    sgtitle(['Osetljivost na početne uslove ' name])
    
    subplot(311)
    plot(times)
    title('Vreme izvršavanja')
    ylabel('Vreme [s]')
    
    subplot(312)
    plot(Niter)
    title('Ukupan broj iteracija algoritma')
    ylabel('Broj iteracija algoritma')
    
    subplot(313)
    plot(accuracies)
    title('Tačnost izvršavanja')
    ylabel('Tačnost [%]')
end

%% Nelinearni evaluator
function evaluate_clustering_performance2(my_fnc, name, N, X)
    Ntest = 10;
    accuracies = zeros(1, Ntest); times = zeros(1, Ntest);
    Niter = zeros(1, Ntest);
    
    L_true = [ones(1, N) 2*ones(1, N)];
    
    for k = 1:Ntest
        Nclass = max(k,2);
        L = randi(Nclass, 1, 2*N);
        tic;
        [Niter(k), L_temp] = my_fnc(X, L);
        %Zbog numerike nekad se desava singularnost cov matrice
        if length(L_temp) ~= length(L)
            L = zeros(1, 1000);
        else
            L = L_temp;
        end

        time = toc; 
        
        modified_L = L;
        for i = 1:2
            chunk = L((i-1)*500 + 1 : i*500);
            median_label = mode(chunk);
            modified_L((i-1)*500 + find(chunk == median_label)) = i;
            modified_L((i-1)*500 + find(chunk ~= median_label)) = median_label;
        end
        L = modified_L;
        
        accuracies(k) = 100*sum(L_true == L) / (2 * N);
        times(k) = time;
    end
    
    figure;
    sgtitle(['Osetljivost na broj klasa ' name])
    
    subplot(311)
    plot(times)
    title('Vreme izvršavanja')
    ylabel('Vreme [s]')
    xlabel('Broj klasa')
    
    subplot(312)
    plot(Niter)
    title('Ukupan broj iteracija algoritma')
    ylabel('Broj iteracija algoritma')
    xlabel('Broj klasa')
    
    subplot(313)
    plot(accuracies)
    title('Tačnost izvršavanja')
    ylabel('Tačnost [%]')
    xlabel('Broj klasa')

    Ntest = 15;
    accuracies = zeros(1, Ntest); times = zeros(1, Ntest);
    Niter = zeros(1, Ntest);
    
    L_true = [ones(1, N) 2*ones(1, N)];
    
    for k = 1:Ntest
        L = randi(2, 1, 2*N);
        tic;
        [Niter(k), L_temp] = my_fnc(X, L);
        %Zbog numerike nekad se desava singularnost cov matrice
        if length(L_temp) ~= length(L)
            L = zeros(1, 1000);
        else
            L = L_temp;
        end
        time = toc; 
        
        modified_L = L;
        for i = 1:2
            chunk = L((i-1)*500 + 1 : i*500);
            median_label = mode(chunk);
            modified_L((i-1)*500 + find(chunk == median_label)) = i;
            modified_L((i-1)*500 + find(chunk ~= median_label)) = median_label;
        end
        L = modified_L;
        
        accuracies(k) = 100*sum(L_true == L) / (2 * N);
        times(k) = time;
    end
    
    display(['Srednje vreme: ' num2str(mean(times)) 's']);
    display(['Srednji broj iteracija: ' num2str(mean(Niter))]);
    display(['Srednja tačnost: ' num2str(mean(accuracies)) '%']);
    
    figure;
    sgtitle(['Osetljivost na početne uslove ' name])
    
    subplot(311)
    plot(times)
    title('Vreme izvršavanja')
    ylabel('Vreme [s]')
    
    subplot(312)
    plot(Niter)
    title('Ukupan broj iteracija algoritma')
    ylabel('Broj iteracija algoritma')
    
    subplot(313)
    plot(accuracies)
    title('Tačnost izvršavanja')
    ylabel('Tačnost [%]')
end