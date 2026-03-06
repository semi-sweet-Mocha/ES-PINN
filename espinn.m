clc
clear

load Std1964.mat
load munsell.mat
load spectralLibrary.mat
sourceFolder = 'D:\20240701\sr2srgb\NTIRE 2020 D65 Val'; 
fileList = dir(fullfile(sourceFolder, '*.mat'));
fileName = fileList(4).name;
filePath = fullfile(sourceFolder, fileName);
data = load(filePath);
if isfield(data, 'srgb')
    image = data.srgb;
else
    error('File %s does not contain variable "srgb".', fileName);
end
numFrames = 5;
imageSequence = cell(numFrames, 1);
vortex = generateVortexPhase(size(image, 1:2), 1);
random = generateRandomPhase(size(image, 1:2));
grating = generateGratingPhase(size(image, 1:2), 20);
% vortexPhase = generateVortexPhase([512, 512], 1);
% randomPhase = generateRandomPhase([512, 512]);
% gratingPhase = generateGratingPhase([512, 512], 50);
% 
% figure;
% subplot(1,3,1);
% imagesc(angle(vortexPhase));
% title('Vortex Phase');
% axis off; 
% % colorbar;
% 
% subplot(1,3,2);
% imagesc(angle(randomPhase));
% title('Random Phase');
% axis off; 
% % colorbar;
% 
% subplot(1,3,3);
% imagesc(angle(gratingPhase));
% title('Grating Phase');
% axis off; 
% % colorbar;
for i = 1:numFrames
    epsilon = 0.01;
    brightnessScale = max(2 - (i-1) / (numFrames-1) * 2, epsilon);
    dimmedImage = im2uint8(image * brightnessScale);
    dimmedImage = im2double(dimmedImage);
    dimmedImage = dimmedImage .* abs(vortex) .* abs(random) .* abs(grating);
    [R, L, ~] = color_constancy(dimmedImage);
    illumination = im2uint8(L);
    imageSequence{i} = illumination;
end
RGB_to_LMS = [0.31399022 0.63951294 0.04649755; 
              0.15537241 0.75789446 0.08670142; 
              0.01775239 0.10944209 0.87256922];
LMS_Total = zeros(size(imageSequence{1}));
weightSum = 0;
for i = 1:numFrames
    imageLMS = im2double(imageSequence{i});
    LMS = reshape(imageLMS, [], 3) * RGB_to_LMS';
    L = mean(LMS(:, 1));
    M = mean(LMS(:, 2));
    S = mean(LMS(:, 3));
    weight = mean([L, M, S]);
    LMS_Total = LMS_Total + weight * imageLMS;
    weightSum = weightSum + weight;
end
LMS_Final = LMS_Total / weightSum;

ref = munsell;
light = model(LMS_Final);
light = light'; 
light= (light/max(light));
TR=light;
TR=diag(TR);
param=[0.3127 0.3290 0.64 0.33 0.30 0.60 0.15 0.06 1];
M = mat_rgb2xyz(param);
XYZ = ref2XYZ(ref,Std,TR,inv(M));
sRGB = XYZ2sRGB(XYZ,inv(M));
XTrain = sRGB;
YTrain = ref;
numTrain = floor(0.8 * size(XTrain, 2));
numVal = size(XTrain, 2) - numTrain;
XVal = XTrain(:, numTrain+1:end);
YVal = YTrain(:, numTrain+1:end);
XTrain = XTrain(:, 1:numTrain);
YTrain = YTrain(:, 1:numTrain);

layers = [
    featureInputLayer(3, 'Normalization', 'none') 
    fullyConnectedLayer(256)                      
    reluLayer                                   
    fullyConnectedLayer(128)                      
    reluLayer                                      
    fullyConnectedLayer(64)                       
    reluLayer                                     
    fullyConnectedLayer(31)                      
    sigmoidLayer                               
];

net = dlnetwork(layers);
alpha = 0;
beta = 0.2; 
averageGrad = [];
averageSqGrad = [];
InitialLearnRate = 0.02; 
LearnRateDropFactor = 0.1; 
t = 0; 
patience = 10; 
bestValLoss = inf;
patienceCounter = 0;
minEpochs = 200; 
numEpochs = 1000;
miniBatchSize = 32;
numObservations = size(XTrain, 2);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);

for epoch = 1:numEpochs
    idx = randperm(numObservations);
    for iteration = 1:numIterationsPerEpoch
        batchIdx = idx((iteration-1)*miniBatchSize + 1:iteration*miniBatchSize);
        XBatch = dlarray(XTrain(:, batchIdx), 'CB');
        YBatch = dlarray(YTrain(:, batchIdx), 'CB');
        [loss, gradients] = dlfeval(@modelGradients, net, XBatch, YBatch, Std, TR, M, alpha, beta);
        t = t + 1;
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, ...
            InitialLearnRate, LearnRateDropFactor);
    end
    XVal_dl = dlarray(XVal, 'CB');
    YVal_dl = dlarray(YVal, 'CB');
    valPred = predict(net, XVal_dl);
    valLoss = customLoss(valPred, YVal_dl, Std, TR, M, alpha, beta);
    if mod(epoch, 5) == 0
        fprintf('Epoch %d, Loss: %.6f, Val Loss: %.6f\n', epoch, loss, valLoss);
    end
    if epoch >= minEpochs  
        if valLoss < bestValLoss
            bestValLoss = valLoss;
            patienceCounter = 0;
        else
            patienceCounter = patienceCounter + 1;
            if patienceCounter >= patience
                fprintf('Early stopping at epoch %d\n', epoch);
                break;
            end
        end
    end
end

% 
% reflectances = reshape(reflectances,[],31);
% reflectances = reflectances';
% XYZ_test = ref2XYZ(reflectances,Std,TR,inv(M));
% sRGB = XYZ2sRGB(XYZ_test,inv(M));
light_test = spectralLibrary.A;
light_test = light_test'; 
light_test= (light_test/max(light_test));
TR=light_test;
TR=diag(TR);
% ref = readmatrix('colorchecker.xlsx','Sheet','Sheet1');
XYZ_test = ref2XYZ(ref,Std,TR,inv(M));
sRGB_test = XYZ2sRGB(XYZ_test,inv(M));
XTest = sRGB_test;
% XTest = imread('D:\NTIRE Val\ARAD_HS_0451_D65.jpg');
% XTest = reshape(XTest,[],3);
% XTest = im2double(XTest);
% XTest = XTest';
% load D:\public_data\NTIRE\NTIRE_2020\NTIRE2020_Validation_Spectral\ARAD_HS_0451.mat
% YTest = reshape(cube,[],31);


XTest = dlarray(XTest, 'CB');
YPred = predict(net, abs(XTest));
YPred = extractdata(YPred);
YPred = abs(double(YPred));
[~,n] = size(XTest);
cc = munsell'; 
r = YPred'; 
RMSE = zeros(n,1);
diff = cc - r;
E = mean(diff);
E1 = mean(abs(diff));

function vortexPhase = generateVortexPhase(shape, charge)
    [X, Y] = meshgrid(linspace(-1, 1, shape(2)), linspace(-1, 1, shape(1)));
    phase = atan2(Y, X) * charge;
    vortexPhase = exp(1i * phase);
end

function randomPhase = generateRandomPhase(shape)
    phase = rand(shape) * 2 * pi;
    randomPhase = exp(1i * phase);
end

function gratingPhase = generateGratingPhase(shape, period)
    [X, Y] = meshgrid(1:shape(2), 1:shape(1));
    phase = sin(2 * pi * (X + Y) / period);
    gratingPhase = exp(1i * phase);
end

function [loss, gradients] = modelGradients(net, X, Y, s, l, M, alpha, beta)
    dlYPred = forward(net, X);
    loss = customLoss(dlYPred, Y, s, l, M, alpha, beta);
    gradients = dlgradient(loss, net.Learnables);
end

function loss = customLoss(dlYPred, Y, s, l, M, alpha, beta)

    data_loss = mean((dlYPred - Y).^2, 'all');
    predArray = extractdata(dlYPred);
    trueArray = extractdata(Y);
    XYZ1 = ref2XYZ(predArray,s,l,inv(M));
    YPred = XYZ2sRGB(XYZ1,inv(M));
    XYZ2 = ref2XYZ(trueArray,s,l,inv(M));
    YTrue = XYZ2sRGB(XYZ2,inv(M));
    pde_residual = mean((YTrue - YPred).^2, 'all'); 
    boundary_loss = mean((XYZ2 - XYZ1).^2, 'all');
    loss = (1 - alpha - beta)*data_loss + alpha * pde_residual + beta * boundary_loss;

end