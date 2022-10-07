function [outputs]=predict(net,inputs)
% inputs: input matice or vector
% net: the trained net
% outputs :the pridected output
inputs=scaledata(inputs,0,1);% data scaling
Number_W=net.NW;% get number of weight matrices
weights=net.IW;% get weights
N1=net.den(1);% get denormalizing values
N2=net.den(2);% get deneormalizing values
for i=1:Number_W
 W=weights(i).F;% load  weights
 %Hidden_layer=logsig(inputs*W);% calculate the hidden layer
 Hidden_layer=logsig(scaledata(inputs*W,N2,N1));% TD: if data is not scaled, large values e.g. 68 always give output of 1 for all points  
 inputs=Hidden_layer;% set the hidden as the input of next hidden layer  
end
outputs=Hidden_layer;% normlized output
outputs=scaledata(outputs,N2,N1);% denormalized output
end