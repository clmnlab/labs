function Distance=extractVdistance(testsubj,total_n)
    subj_dat=load_Psession(testsubj,total_n);
    totalDistanceL0=cell(1,total_n);
    totalDistanceR0=cell(1,total_n);
    for session_n = 1:total_n
        subj_session=subj_dat{session_n};
        totalDistanceL0{session_n}=cal_distance(subj_session.learnL);
        totalDistanceR0{session_n}=cal_distance(subj_session.learnR);
    end
    totalDistanceL=cell2mat(totalDistanceL0);
    totalDistanceR=cell2mat(totalDistanceR0);
    totalDistance=nan(1,length(totalDistanceL)+length(totalDistanceR));
    distance_l=1;
    distance_r=1;
    for trials_n = 1:length(totalDistance)
        if rem(trials_n,2)==1
            totalDistance(trials_n)=totalDistanceR(distance_r);
            distance_r=distance_r+1;
        elseif rem(trials_n,2)==0
            totalDistance(trials_n)=totalDistanceL(distance_l);
            distance_l=distance_l+1;
        end
    end
    Distance.total=totalDistance;
    Distance.left=totalDistanceL;
    Distance.right=totalDistanceR;
end

function distance_trial=cal_distance(test)
    distance_trial=nan(1,length(test.XY));
    for trial_n = 1:length(distance_trial)
        distance_trial(trial_n)=mov_distance(test.XY{trial_n});
    end
end
function distance_sum=mov_distance(testXY)
    distance_sum=0;
    for i=2:length(testXY)
        distance_sum=distance_sum+norm(testXY(:,i)-testXY(:,i-1));
    end
end