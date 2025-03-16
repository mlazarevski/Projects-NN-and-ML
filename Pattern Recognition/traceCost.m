function J3 = traceCost(X, L, maxClass)
    
    N = length(X);
    P = zeros(1,maxClass); M = zeros(2,maxClass); S = zeros(2,2*maxClass);
    Sw = 0; Sb = 0;
    for k = 1:maxClass
        idx = (L == k);
        M(:,k) = mean(X(:, idx), 2);
        S(:,2*k-1:2*k) = cov(X(:, idx)');
        P(k) = sum(idx)/N;
        Sw = Sw + S(:,2*k-1:2*k).*P(k);
    end
    
    M0 = sum(M.*P, 2);
    
    for k = 1:maxClass
        Sb = Sb + P(k)*(M(:,k)-M0)*(M(:,k)-M0)';
    end
    
    J3 = trace(pinv(Sw+Sb)*Sw);
end