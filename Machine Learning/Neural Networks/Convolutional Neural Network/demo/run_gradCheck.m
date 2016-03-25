
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

numLayers=4;
numClasses=10;
layers=cell(numLayers+1,1);

layers{1}=make_convLayer_struct('tanh',[28,28,1],[9,9,1,2],'none',[]);
layers{2}=make_convLayer_struct('tanh',layers{1}.outDim,[5,5,2,4],'max',[4,4]);
layers{3}=make_hiddenLayer_struct('tanh',30,prod(layers{2}.outDim));
layers{4}=make_outLayer_struct(10,layers{3}.outDim);
layers=initParams(layers);
fprintf('Displaying network for gradient check.....\n');
display_net(layers);

theta=layers{end}.theta;

fprintf('Checking gradient....');

N=5;
f=@(param) cnn_cost(x_train(:,:,:,1:N),y_train(1:N),layers,param,false);
num_grad=computeNumericalGradient(f,theta);


[~,grad]=cnn_cost(x_train(:,:,:,1:N),y_train(1:N),layers,theta);

diff=norm(num_grad-grad,2);
fprintf('Error:%d\n',diff);






