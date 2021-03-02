function [ Idx, K_rep, Jcluster ] = kmeans( numImages, numClusters )
% numImages - Input of how many images we are dealing with
% numCluster - Input of how many clusters we will be dividing
% Idx - Output that a list of N group assignment indices (c1, c2, ... ,cn)
% K_rep - Output that K group representatives
% Jcluster - Output that converged value of each iteration

% figuring out how many images we have
numDataPts = size(numImages, 1);

% randomly assigning K data points as the K group representatives
% this is the way how I choose a method for initializing before next run
temp = randperm(numDataPts, numClusters);
K_rep = numImages(temp, :);

%initializing variables
Idx = zeros(1, numDataPts);
iteration = 1;
Jcluster = 0;
% J_array is for plotting Jcluster over each iteration
J_array = zeros(1); %initializing an array

% This is for comparing before and after values of K_rep, and if those are
% the same, terminate the loop, which is converged.
compare = K_rep;
% I got an idea to construct the structures below from the github, link is:
% https://github.com/matzewolf/kMeans/blob/master/kMeans.m
while 1

    NormSqSum = 0;
    
    for i = 1 : numDataPts
        distance = zeros(1, numClusters);
        
        for j = 1 : numClusters
            distance(j) = norm((numImages(i,:) - K_rep(j,:)));
        end
        
        % Determine c_i, which is a group index
        [~,cluster] = min(distance);
        Idx(i) = cluster;
        
        NormSqSum = NormSqSum + min((distance).^2);
        
    end
    
    for l = 1 : numClusters
        K_rep(l,:) = mean(numImages(Idx==l,:),1);
    end
    
    % Termination point
    if K_rep == compare
        break;
    else
        compare = K_rep;
    end
    
    Jcluster = double((1/numDataPts)) * NormSqSum;
    % Create an array of Jcluster values for plotting Jcluster
    J_array(iteration, 1) = Jcluster;
    
    iteration = iteration + 1;
end
J_array(iteration, 1) = Jcluster;

% make a file of J_array to use it in the main function
save('iteration&J','J_array');
fprintf('%d iterations used', iteration);
fprintf(' and J cluster is %f  \n', Jcluster);

end
