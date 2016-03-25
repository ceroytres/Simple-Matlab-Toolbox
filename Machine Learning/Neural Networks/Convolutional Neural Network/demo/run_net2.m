close all; clc; clear all;
load('train_set.mat');
load('test_set.mat');

y_train=double(y_train);
y_train(y_train==0)=10;
y_test=double(y_test);
y_test(y_test==0)=10;

%Equivalent im2double
x_train=double(x_train)/255;
x_test=double(x_test)/255;



% Zero mean data
mu=mean(x_train,3);
x_train=bsxfun(@minus,x_train,mu);
x_test=bsxfun(@minus,x_test,mu);

%Reshape data
x_train=reshape(x_train,28,28,1,[]);
x_test=reshape(x_test,28,28,1,[]);


    
numLayers=2;
numClasses=10;


layers=cell(numLayers+1,1);
layers{1}=make_convLayer_struct('relu',[28,28,1],[9,9,1,10],'max',[2,2]);
layers{2}=make_outLayer_struct(numClasses,prod(layers{1}.outDim));

layers=initParams(layers);
fprintf('Displaying network.....\n');
display_net(layers);

theta=layers{end}.theta;

fObj=@(theta,x,y) cnn_cost(x,y,layers,theta);
options.epochs = 1;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;

tic;
[theta_opt,J]=minFuncSGD(fObj,theta,x_train,y_train,options);
toc;


plot(J,'linewidth',3);
ylabel('Batch Cost $J(\theta)$','fontsize',12);
xlabel('Batch Number','fontsize',12);
title('Cross Entropy Cost on Architecture 2','fontsize',12);



[~,~,pred]=cnn_cost(x_test,[],layers,theta_opt,true);
acc=mean(pred'==y_test);
fprintf('Accuracy:%f',acc);