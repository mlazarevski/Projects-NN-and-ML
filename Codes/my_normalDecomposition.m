function [Niter, L] = my_normalDecomposition(X, L)
    J2 = @(X, M, S) log(det(S)) + sum(((X-M)'*inv(S)).*(X-M)',2); %Pi = const(i), a mnozenje i sabiranje sa const, ne utice na dJ
    N = length(X);
    
    d = zeros(4, N);
    J =[];
    Niter = 0;
    for iter = 1:1000
        Niter =  Niter + 1;
        M = zeros(2,4); S = zeros(2,8);
        for k = 1:4
            M(:,k) = mean(X(:, L == k), 2);
            S(:,2*k-1:2*k) = cov(X(:, L == k)');
        end
        for k = 1:4
            d(k,:) = J2(X, M(:,k), S(:,2*k-1:2*k)); 
        end
        
        [value, L] = min(d);
        
        J(iter) = sum(value);
        if (iter > 1) && (J(iter)*1.005 > J(iter-1))
            tolerance = tolerance - 1;
            if tolerance == 0
                break
            end
        else
            tolerance = 10;
        end
    end
end