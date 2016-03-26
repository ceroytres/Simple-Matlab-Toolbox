function average_error = grad_check(fun, theta0,delta, num_checks, varargin)

  sum_error=0;

  N=length(theta0);
  
  idx=randperm(N);
  
  if num_checks>N
      num_checks=N;
  end
  

  for i=1:num_checks
    T = theta0;
    j = idx(i);
    T0=T; T0(j) = T0(j)-delta;
    T1=T; T1(j) = T1(j)+delta;

    [f,g] = fun(T, varargin{:});
    f0 = fun(T0, varargin{:});
    f1 = fun(T1, varargin{:});

    g_est = (f1-f0) / (2*delta);
    error = abs(g(j) - g_est);
    sum_error = sum_error + error;
    
    %fprintf('Iter:%d, i:%d, err:%g, g_est:%f, g:%f f:%f\n',i,j,error,g_est,g(j),f);
    
  end

  average_error=sum_error/num_checks;
  fprintf('Avg. Error:%g\n',average_error);
end
