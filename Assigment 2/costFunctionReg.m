function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


 A = sigmoid (X*theta) ;
       for i=1:m
          J = J -log( A(i) )*y(i) -(1-y(i))*(log(1-A(i)))  ;
       end
       
       for i=2:size(theta,1)
       J= J+lambda*theta(i)^2/2;  
       end 
   
J=J/m;
       
      A=A-y;
      for i=1:size(grad,1)
      grad(i)=sum(A.*X(:,i));
      end
      grad=grad/m;
      grad=grad+lambda/m*(theta);
      grad(1)=grad(1)-lambda/m*(theta(1));



% =============================================================

end
