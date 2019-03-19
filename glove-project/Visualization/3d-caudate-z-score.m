caudate_index = 2;

mask_map = load_untouch_nii('mask_resample_stats.group.rwdtime.GAGB_n25.run1to3.nii.gz');
caudate_volume = (mask_map.img(:, :, :, 1, 1) ~= 0);
%caudate_mesh = isosurface(smooth3(caudate_volume, 'box', 3), 0.5);
caudate_mesh = isosurface(caudate_volume, 0.5);

brain_map = load_untouch_nii('CIT168toMNI152-FSL_T1w_brain.nii.gz');
brain_volume = (brain_map.img > 1);
brain_mesh = isosurface(smooth3(brain_volume, 'box', 11), 0.5);

stat_map = load_untouch_nii('resample_stats.group.rwdtime.GAGB_n25.run1to3.nii.gz');
stat_volume = stat_map.img(:, :, :, 1, 2);

[num_ver, ~] = size(caudate_mesh.vertices);
color_map = zeros(num_ver, 1);

for i = 1:num_ver
    vertex = round(caudate_mesh.vertices(i, :));
    color_map(i) = stat_volume(vertex(2), vertex(1), vertex(3));
end

color_map = NNsmooth(caudate_mesh.vertices, caudate_mesh.faces, color_map', 2);

ax1 = axes;

p2 = patch('Vertices', brain_mesh.vertices, ...
      'Faces', brain_mesh.faces, ...
      'FaceColor', 'white', ...
      'FaceAlpha', .3, ...
      'EdgeColor', 'none', ...
      'AmbientStrength', 1, ...
      'BackFaceLighting', 'unlit');
  
set(gca,'Color','k')

camlight;
lighting gouraud;

xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis');

daspect([1 1 1]);
view(3); 
axis ij;
axis equal;
ax_hidden = axes('Visible', 'off', 'hittest', 'off');

p1 = patch('Vertices', caudate_mesh.vertices, ...
      'Faces', caudate_mesh.faces, ...
      'FaceColor', 'interp', ...
      'FaceVertexCData', color_map', ...
      'EdgeColor', 'none', ...
      'DiffuseStrength', 0, ...
      'SpecularStrength', 0, ...
      'AmbientStrength', 1.0, ...
      'FaceLighting', 'flat');
  
camlight;
lighting gouraud;
colormap(flipud(cbrewer('div', 'RdBu', 255)));
caxis([-6 6]);

xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis');

daspect([1 1 1]);
view(3); 
axis ij;
axis equal;
colorbar;

set(p1, 'Parent', ax_hidden);

linkprop([ax1 ax_hidden], {'CameraPosition', 'XLim', 'YLim', 'ZLim', 'Position', 'CameraUpVector'});
