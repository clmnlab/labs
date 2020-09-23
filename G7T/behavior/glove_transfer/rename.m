txtlist=dir('*txt');
for txt_n = 1:length(txtlist)
    intxt=txtlist(txt_n).name;
    outtxt=replace(intxt,'Reward','Hit');
    inputfile=strcat(txtlist(txt_n).folder,'/',intxt);
    outputfile=strcat(txtlist(txt_n).folder,'/',outtxt);
    movefile(inputfile,outputfile);
end