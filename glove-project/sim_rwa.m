% Free parameters: gamma (decaying rate), tau (sensitivity of reward to error)
load('GA01-Cali');
tar = [160 -160]; tau = 10; gam = 0.95;
N = 1000; mu_glove = mData; sig{1} = corr(allData');  
G = [420 420];
for j=1:N  % Number of trials
 	if j<=2 % Wait until 2 trials
 		mu(j,:) = mu_glove'; % Initialize mean
		sig{j} = sig{1}; % Initialize variance
      else     % From the 3rd trial
       	sumR = sum((gam.^[j-2:-1:0]).*rew(1:j-1));  % Sum rewards
        	mu(j,:) = ((gam.^[j-2:-1:0]).*rew(1:j-1))*u(1:j-1,:)/sumR; % Update mean
        	sig{j} = rexpl(1:j-1,:)'*expl(1:j-1,:)/sumR; % Update variance
	end
    	u(j,:) = mvnrnd(mu(j,:),sig{j});  % Select action from updated policy    
    	expl(j,:) = u(j,:)-mu(j,:);              % Calculate exploration
    	XY(j,:) = G.*(u(j,:)*A);                          % Map to cursor position
    	err(j) = norm(XY(j,:)-tar);            % Calculate error
    	rew(j) = rewfunc(err(j),tau);       % Calculate reward for the error
    if j==1 rexpl(1,:) = rew(j)*expl(j,:); end
    rexpl(1:j,:) = [gam*rexpl(1:j-1,:); rew(j)*expl(j,:)]; % Reward-weighted exploration
end

figure;
subplot(131); plot(err); 
subplot(132); plot(rew);
subplot(133); for j=1:N mexpl(j) = trace(sig{j});end
plot(mexpl);