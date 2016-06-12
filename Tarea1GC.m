% Tarea n°1 para el electivo profesional de Computación Evolutiva
% Primer semestre 2016
% Universidad de Santiago de Chile
% Autores: Pablo Salinas Cabañas, Claudia Guzman
% Fecha: 1 de Junio de 2016
% 
function Feat_Index =  Tarea1GC
%% Importar datasets

clear
clc
close all


global GC
gece =  load('gece.data','-mat');
GC=gece.GC;

%% Parametros

global size_poblacion
size_poblacion=80;%Tamaño poblacion inicial
tasa_cruzamiento=0.5;%Tasa de cruzamiento
tasa_mutacion=0.3;%Tasa de mutación
global size_string
size_string=24;%Largo del string
generaciones=40;%Cantidad de generaciones
size_torneo = 4; %Tamaño torneo
poblacion_elite=4; %Tamaño de la población de elite
poblaciones_sin_mejora=20; %Cantidad de poblaciones aceptadas sin mejoras
global lambda
lambda=0.1;
tiempo_inicio = cputime;
parametros = gaoptimset('CreationFcn', {@poblacionInicial},...
                     'PopulationSize',size_poblacion,...
                     'Generations',generaciones,...
                     'PopulationType', 'bitstring',... 
                     'SelectionFcn',{@selectiontournament,size_torneo},...
                     'MutationFcn',{@mutationuniform, tasa_mutacion},...
                     'CrossoverFcn', {@crossoverarithmetic,tasa_cruzamiento},...
                     'EliteCount',poblacion_elite,...
                     'StallGenLimit',poblaciones_sin_mejora,...
                     'PlotFcns',{@gaplotbestf},...  
                     'Display', 'iter'); 
rand('seed',1)

%% Algoritmo genético

[chromosome,~,~,~,~,~] = ga(@Fitness,size_string,parametros);
% Tiempo de ejecucion
total = cputime - tiempo_inicio;
fprintf('Tiempo de ejecucion: %i',total);

Best_chromosome = chromosome; % Best Chromosome
Feat_Index = find(Best_chromosome==1); % Index of Chromosome

end


%% Función de creación de la primera generación
function [poblacion] = poblacionInicial(string_size,~,options)
    RD = rand;  
    poblacion = (rand(options.PopulationSize, string_size)> RD);
%    fprintf('Población inicial\n\n');
%     for i=1:options.PopulationSize
%         for j=1:string_size
%             fprintf('%1.0f ',poblacion(i,j));
%         end
%         fprintf('\n');
%     end
%     fprintf('\n');
end

%% Función de fitness
function [valor_fitness] = Fitness(string)
    %% Paso del string a lista de atributos seleccionados
    global size_string
    global lambda
    listaAux=zeros(1,size_string);
    contador=0;
    for i=1:size_string
        if(string(i)==1)
            contador=contador+1;
            listaAux(1,contador)=i;
        end
    end
    lista_columnas=listaAux(listaAux~=0);
    %% SVM
    if (contador>0)
        global GC
        columnasSeleccionadas=GC(1:end,lista_columnas);
        clasificaciones=GC(1:end,25);

        p=0.5;
        [Train,Test] = crossvalind('HoldOut', clasificaciones,p);

        trainingSample=columnasSeleccionadas(Train,:);
        trainingLabel=clasificaciones(Train,1);
        testingSample=columnasSeleccionadas(Test,:);
        testingLabel=clasificaciones(Test,1);

        svmStruct =svmtrain(trainingSample,trainingLabel,...
            'showplot',false,'kernel_function','rbf','rbf_sigma',0.5);

        outLabel = svmclassify(svmStruct,testingSample,'showplot',false);

        valor_fitness=(1-lambda)*(1-(sum(grp2idx(outLabel)==grp2idx(testingLabel))/sum(Test)))+lambda*(contador/size_string);
    else
        valor_fitness=1;
    end
end
