#!/bin/tcsh

foreach i (`count -digit 3 1 280`)
	set temp = fan.roi.GA.$i.nii.gz
	if ( ! -e $temp) then
		echo "$temp doesn't exist"
	endif
end
