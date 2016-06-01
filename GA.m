function Feat_Index =  Binary_Genetic_Algorithm_Hezy_2013
% Written by BABATUNDE Oluleye H, PhD Student

% Address: eAgriculture Research Group, School of Computer and Security
% Science, Edith Cowan University, Mt Lawley, 6050, WA, Australia
% Date:  2013
% Please cite any of the article below (if you use the code), thank you

%  "BABATUNDE Oluleye, ARMSTRONG Leisa J, LENG Jinsong and DIEPEVEEN Dean (2014). 
%  Zernike Moments and Genetic Algorithm: Tutorial and APPLICATION. 
%  British Journal of Mathematics & Computer Science. 
%  4(15):2217-2236."

%%% OR
%BABATUNDE, Oluleye and ARMSTRONG, Leisa and LENG, Jinsong and DIEPEVEEN (2014).
% A Genetic Algorithm-Based Feature Selection. International Journal 
%of Electronics Communication and Computer Engineering: 5(4);889--905.

% DataSet here
%Ionosphere dataset from the UCI machine learning repository:                   
%http://archive.ics.uci.edu/ml/datasets/Ionosphere                              
%X is a 351x34 real-valued matrix of predictors. Y is a categorical response:   
%"b" for bad radar returns and "g" for good radar returns.                      

% NOTE: You can run this code directory on your PC as the dataset is
% available in MATLAB software

clear all
global Data 
% load ionosphere.mat  % This contains X (Features field) and Y (Class Information)
Data  = load('ionosphere.mat'); % This is available in Mathworks
GenomeLength =30; % This is the number of features in the dataset
tournamentSize = 2;
options = gaoptimset('CreationFcn', {@PopFunction},...
                     'PopulationSize',50,...
                     'Generations',100,...
                     'PopulationType', 'bitstring',... 
                     'SelectionFcn',{@selectiontournament,tournamentSize},...
                     'MutationFcn',{@mutationuniform, 0.1},...
                     'CrossoverFcn', {@crossoverarithmetic,0.8},...
                     'EliteCount',2,...
                     'StallGenLimit',100,...
                     'PlotFcns',{@gaplotbestf},...  
                     'Display', 'iter'); 
rand('seed',1)
nVars = 14; % 
FitnessFcn = @FitFunc_KNN; 
[chromosome,~,~,~,~,~] = ga(FitnessFcn,nVars,options);
Best_chromosome = chromosome; % Best Chromosome
Feat_Index = find(Best_chromosome==1); % Index of Chromosome
end


function [SVM]=SVM_execute(X1,Y1)
    SVM=ClassificationKNN.fit(X1,Y1,'NSMethod','exhaustive','Distance','euclidean');
end


%%% POPULATION FUNCTION
function [pop] = PopFunction(GenomeLength,~,options)
RD = rand;  
pop = (rand(options.PopulationSize, GenomeLength)> RD); % Initial Population
end

%%% FITNESS FUNCTION   You may design your own fitness function here
function [FitVal] = FitFunc_KNN(pop)
global Data
FeatIndex = find(pop==1); %Feature Index
X1 = Data.X;% Features Set
Y1 = grp2idx(Data.Y);% Class Information
X1 = X1(:,[FeatIndex]);
NumFeat = numel(FeatIndex);
Compute = SVM_execute(X1,Y1);
Compute.NumNeighbors = 3; % kNN = 3
FitVal = resubLoss(Compute)/(34-NumFeat);
end
