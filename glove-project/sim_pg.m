load('GA01-Cali');
tar = [160 -160]; tau = 10; alpha = 0.1;
N = 3000; mu_glove = mData; sig{1} = corr(allData');  G = [420 420];
% Free parameters: alpha (learning rate), tau (sensitivity of reward to error)
    base(1) = 0.5; % Initialize baseline
    for j=1:N  % Number of trials
	    if j==1
	        mu(j,:) = mu_glove';   % Initialize mean
	        sig{j} = sig{1};                % Initialize variance
	    else
	        mu(j,:) = mu(j-1,:) + alpha*expl(j-1,:)*(rew(j-1)-base(j-1));  % Update mean
	        sig{j} = sig{j-1} + alpha* (expl(j-1,:)'*expl(j-1,:)-sig{j-1});   % Update variance
	    end
	    u(j,:) = mvnrnd(mu(j,:),sig{j});  % Select action from update policy    
	    expl(j,:) = u(j,:)-mu(j,:);   % Calculate exploration 
    	XY(j,:) = G.*(u(j,:)*A);                          % Map to cursor position
	    err(j) = norm(XY(j,:)-tar); % Calculate error
    	    rew(j) = rewfunc(err(j),tau); % Calculate reward for the error
    	    base(j) = mean(rew(1:j)); % Update baseline
    end
figure;
subplot(131); plot(err); hline(40);
subplot(132); plot(rew);
subplot(133); for j=1:N mexpl(j) = trace(sig{j});end
plot(mexpl);
