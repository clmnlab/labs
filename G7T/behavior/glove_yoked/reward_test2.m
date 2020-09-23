close all
clearvars
folname='main_ex';
subID=strcat(folname,'/GT_Main_1/GT_Main_1');
sessions=load_session(subID,5);
stim1_pos=sessions{1}.stim_ex1;
stim2_pos=sessions{1}.stim_ex2;
radius=100;
inframe=sessions{1}.hitstim_Frames;
test_seq1=sessions{1}.test_seq;
test_seq2=sessions{1}.test_seq2;
ifi=sessions{1}.ifi;
session_n=5;
inputsession=sessions{session_n};

total_reward=nan(1,20);
for trial_n=1:length(inputsession.total_XY)
    incount=0;
    reward=0;
    stim_n=1;
    XY=inputsession.total_XY{trial_n};
    for i = 1:length(XY)
        switch rem(trial_n,2)
            case 1
                stimPos=stim1_pos;
                testSeq=test_seq1;
            case 0
                stimPos=stim2_pos;
                testSeq=test_seq2;
        end
        if IsInRect(XY(1,i),XY(2,i),stimPos(:,str2double(testSeq(stim_n))))
            incount=incount+1;
            if incount==inframe
                reward=reward+1;
                stim_n=stim_n+1;
                incount=0;
            end
        else
            incount=0;
        end
        
        if stim_n>length(test_seq1)
            stim_n=1;
        end
    end
    total_reward(trial_n)=reward;
end

function sessions=load_session(subID,total_n)
    sessions=cell(1,total_n);
    for i = 1:total_n
        sessions{i}=load(strcat(subID,'_',num2str(i),'.mat'));
    end
end
