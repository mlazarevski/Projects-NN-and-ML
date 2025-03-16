function [Niter, L] = my_MaxLikelihoodClustering(X, L)

    numClasses = length(unique(L));
    N = length(X); Niter = 0;
    
    P = zeros(1,numClasses); M = zeros(2,numClasses); S = zeros(2,2*numClasses);
    for k = 1:numClasses
        idx = (L == k);
        M(:,k) = mean(X(:, idx), 2);
        S(:,2*k-1:2*k) = cov(X(:, idx)');
        P(k) = sum(idx)/N;
    end
    
    qprev = zeros(numClasses, N);  q = zeros(numClasses, N);
    for iter = 1:10000
        Niter = iter;
        for i = 1:numClasses
            q(i, :) = P(i) * mvnpdf(X', M(:, i)', S(:, (2*i-1):(2*i)))';
        end
        q = q ./ sum(q);
        
        if max(max(abs(qprev-q))) < 1e-4 
            break
        end
        
        P = sum(q,2)/N;
    
        for i = 1:numClasses 
            M(:,i) = (X*q(i,:)'./P(i)/N)';
            S(:,2*i-1:2*i) = q(i,:).*(X-M(:,i))*(X-M(:,i))'/P(i)/N;
        end
    
        qprev = q;
    end
    [~, L] = max(q);
end