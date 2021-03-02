
load mnist.mat;

% Determine how many points we have and what dimensions are the points
Datapoints = size(trainX, 1);
Dimensions = size(trainX, 2);

% Let K and P
K = 10; P = 1;
% K = 10; P = 20;
% K = 5; P = 10;

% Initializing variables
J = zeros(1, P); % an array of Jcluster values of each run

% minK_rep & maxK_rep are the k-group representatives of the minimum and
% the maximum Jcluster values after P runs, respectively
minK_rep = zeros(K, Dimensions);
maxK_rep = zeros(K, Dimensions);
minJ = 0; maxJ = 0;
bestArray = zeros(1);
worstArray = zeros(1);

for a = 1:P
    % for the input trainX, devide it by 255 and convert it to 'double'
    % values in order to make Jcluster small as shown in the textbook
    [ Idx, K_rep, Jcluster ] = kmeans( double(trainX)/255, K );
    % we load the file 'iteration&J.mat' for using updated J_array
    load iteration&J.mat;
    
    % updating maximum and minimum value of Jcluster for each run
    J(a) = Jcluster;
    minJ = min(J); maxJ = max(J);
    
    if minJ == J(a)
        minK_rep = K_rep;
        bestArray = J_array;
    end
    if maxJ == J(a)
        maxK_rep = K_rep;
        worstArray = J_array;
    end
    
    % Show how many number of run is being processed
    fprintf('%d run\n', a);
end
%% Plots K-group representatives of the maximum and the minimum Jcluster

% To plot the data sets as images, we need to reshape it as 28 * 28 matrix
figure(1); hold on
for i=1:K
   subplot(2,5,i)
   imshow(reshape(minK_rep(i,:), [28,28])');
end
hold off

figure(2); hold on
for i=1:K
    subplot(2,5,i)
    imshow(reshape(maxK_rep(i,:), [28,28])');
end
hold off

%% Plotting the maximum and the minimum Jcluster over number of iterations

figure(3)
%subplot(1,2,1)
x1 = size(bestArray);
scatter(1:x1, bestArray)
title('best cluster')
figure(4)
%subplot(1,2,2)
x2 = size(worstArray);
scatter(1:x2, worstArray)
title('worst cluster')

%% nearest 10 data points

% For right usage of units, we convert all integer values of trainX to
% double values
doubleX = double(trainX)/255;
x = size(doubleX, 1); % x indicates how many points in doubleX

% initializing variables
dist1 = zeros(K, x);
dist2 = zeros(K, x);
nearPoints_best = zeros(10*K, Dimensions);
nearPoints_worst = zeros(10*K, Dimensions);

% To get 10 nearest data points, we need to calculate all the distance
% between each K-group representatives of the best and worst clusters
% and all the points in the data set.
for i = 1:x
    
    for j = 1:K
        dist1(j,i) = norm(doubleX(i,:) - minK_rep(j,:));
        dist2(j,i) = norm(doubleX(i,:) - maxK_rep(j,:));
    end
end

% After calculation, we figure out the nearest 10 points of both the best and
% the worst cases. After that, stack those points into one array.
for m = 1:K
    [~,cluster1] = mink(dist1(m,:), 10);
    [~,cluster2] = maxk(dist2(m,:), 10);
        
    nearPoints_best((10*(m-1)+1):(10*m), :) = trainX(cluster1,:);
    nearPoints_worst((10*(m-1)+1):(10*m), :) = trainX(cluster2,:);
        
end    

%% Plotting nearests 10 data points
for i = 1:K
    figure(i); hold on
    % we can change j values to see different pictures of a different
    % k-group representatives
    for j = (10*(i-1)+1) : (10*i)%j = 81:90 %        
        subplot(2,5,j-10*(i-1))%j-80))
        imshow(reshape(nearPoints_best(j,:), [28,28])');
    end
    hold off
end
