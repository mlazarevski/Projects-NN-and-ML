function [Niter, L] = my_cMeans(X, L)
    C = length(unique(L));
    J1 = @(X, M) sum(abs(sqrt(X - M)));
    N = length(X);
    
    d = zeros(C, N);
    J =[];
    Niter = 0;
    for iter = 1:1000
        Niter = iter;
        M = zeros(2,C);
        for k = 1:C
            M(:,k) = mean(X(:, L == k), 2);
        end
    
        for k = 1:C
            d(k,:) = J1(X, M(:,k)); %razdaljina X od centra Mi
        end
        
        [value, L] = min(d);
        
        J(iter) = sum(value);
        if (iter > 1) && (J(iter)*1.005 > J(iter-1))
            tolerance = tolerance - 1;
            if tolerance == 0
                break
            end
        else
            tolerance = 100;
        end
    end
end