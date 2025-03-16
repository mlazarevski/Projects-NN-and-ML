function [L, Y] = my_BranchAndBound(X, L)
    Nclass = length(unique(L));
    newBranch.L = 1;
    newBranch.Jnew = 0;
    newBranch.Jold = 0;
    currentLeaves = newBranch;
    
    permIndices = randperm(size(X, 2));
    Y = X(:, permIndices);
    
    for n=2:length(Y)
        previousLeaves = currentLeaves;
        currentLeaves = struct('L', cell(1, 0), 'Jnew', [], 'Jold', []);
        for Branch = previousLeaves
            maxClass = min(numel(unique(Branch.L))+1,Nclass);
            for k = 1:maxClass
                newBranch.L = [Branch.L k];
                newBranch.Jnew = traceCost(Y(:,1:n), newBranch.L, min(numel(unique(newBranch.L)),Nclass));
                newBranch.Jold = Branch.Jnew;
                currentLeaves = [currentLeaves, newBranch];
            end
        end
        
        [Jmin, idx] = min([currentLeaves.Jnew]);
        newBranch = currentLeaves(idx);
        currentLeaves = currentLeaves([currentLeaves.Jold] <= Jmin);
        if isempty(currentLeaves)
                    currentLeaves = newBranch;
        end
    end
    
    [~, idx] = min([currentLeaves.Jnew]);
    finalBranch = currentLeaves(idx);
    L = finalBranch.L;
end