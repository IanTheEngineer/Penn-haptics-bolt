
object_list = process_h5_file( '/home/imcmahon/Desktop/Desktop/car_sponge_101_all.h5', true);

%for i = 1:numel(object_list,2)
i = 1;
object = object_list(i);

%% Constants
center_gripper_state = 7;
disabled_state = 0;
thermal_hold_state = 1;
slow_slide_state = 2;
squeeze_state = 3;
tap_state = 5;
done_state = 4;
fast_slide_state = 6;

LEFT = 2;
RIGHT = 1;

sample_freq = 100; %Hz

ylabel_font_size = 18;
ylabel_position = -2.5;

figure_width = 1200;
aspect_ratio = 2.5/1;
plot_line_width = 1;


controller_states = object.data.controller_state.controller_state;

center_grip_ind = find(controller_states == center_gripper_state, 1, 'first');

done_ind = find(controller_states == done_state, 1, 'first');


center_grip_resize_ind = (center_grip_ind - center_grip_ind)/sample_freq;
done_resize_ind = (done_ind - center_grip_ind)/sample_freq;

sample_time_sec = 1/sample_freq;
relevant_x = [center_grip_resize_ind:sample_time_sec:done_resize_ind];
start_ind = center_grip_ind;
end_ind = done_ind;


center_list = [center_grip_resize_ind center_grip_resize_ind];
done_list = [done_resize_ind done_resize_ind];

tap_ind = (find(controller_states == tap_state, 1, 'first')-center_grip_ind)/sample_freq;
tap_list = [tap_ind tap_ind];
thermal_ind = (find(controller_states == thermal_hold_state, 1, 'first')-center_grip_ind)/sample_freq;
thermal_list = [thermal_ind thermal_ind];
squeeze_ind = (find(controller_states == squeeze_state, 1, 'first')-center_grip_ind)/sample_freq;
squeeze_list = [squeeze_ind squeeze_ind];
slow_slide_ind = (find(controller_states == slow_slide_state, 1, 'first')-center_grip_ind)/sample_freq;
slow_slide_list = [slow_slide_ind slow_slide_ind];
fast_slide_ind = (find(controller_states == fast_slide_state, 1, 'first')-center_grip_ind)/sample_freq;
fast_slide_list = [fast_slide_ind fast_slide_ind];



h = figure;
hold on;
set(h, 'position', [1921, 200, figure_width, figure_width/aspect_ratio])


%% First Subplot - Gripper Aperture
ax(1) = subplot(7,1,1);
plot(relevant_x, object.data.gripper_aperture.joint_position(start_ind:end_ind), 'Color', [18 90 215]/255, 'LineWidth',plot_line_width);
%Ylabel Alignment
y = ylabel('$X_g\ _{(m)}$','Interpreter','latex','FontSize',ylabel_font_size, 'VerticalAlignment','middle');
set(get(gca,'YLabel'),'Rotation',0.0);
current_ylabel_pos = get(y,'pos'); % Read position [x y z] 
top_plot_ylabel_x = current_ylabel_pos(1) + ylabel_position;
set(y,'pos',[top_plot_ylabel_x, current_ylabel_pos(2), current_ylabel_pos(3)]) 
set(gca,'XTickLabel', '');
%Dotted Lines
line(tap_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(thermal_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(squeeze_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(slow_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(fast_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
box off;

%Top Subplot Only: Text above all axes
ymax = max(ylim);
str1 = {'Center'};
text( (tap_ind + center_grip_resize_ind)/2 ,ymax*1.2, str1,'HorizontalAlignment','center', 'FontAngle', 'italic');
str1_5 = {'Tap'};
text( (squeeze_ind + tap_ind)/2 ,ymax*1.2, str1_5,'HorizontalAlignment','center', 'FontAngle', 'italic');
str2 = {'Squeeze'};
text( (thermal_ind + squeeze_ind)/2 ,ymax*1.2, str2,'HorizontalAlignment','center', 'FontAngle', 'italic');
str3 = {'Static Hold'};
text( (slow_slide_ind + thermal_ind)/2 ,ymax*1.2, str3,'HorizontalAlignment','center', 'FontAngle', 'italic');
str4 = {'Slow Slide'};
text( (fast_slide_ind + slow_slide_ind)/2 ,ymax*1.2, str4,'HorizontalAlignment','center', 'FontAngle', 'italic');
str5 = {'Fast Slide'};
text( (done_resize_ind + fast_slide_ind)/2 ,ymax*1.2, str5,'HorizontalAlignment','center', 'FontAngle', 'italic');

%% Second Subplot - Z Transform
ax(2) = subplot(7,1,2);
hold on;
z_tf = object.data.transforms.translation(3,:);
non_zero_tf_idx = find(z_tf, 1, 'first');
z_tf(1:non_zero_tf_idx-1) = z_tf(non_zero_tf_idx);

plot(relevant_x,z_tf(start_ind:end_ind), 'Color', [10 156 63]/255, 'LineWidth',plot_line_width);
y = ylabel('$Z_{tf}\ _{(m)}$ ','Interpreter','latex','FontSize',ylabel_font_size, 'VerticalAlignment','middle');
set(get(gca,'YLabel'),'Rotation',0.0);
set(gca,'XTickLabel', '');
current_ylabel_pos = get(y,'pos'); % Read position [x y z] 
set(y,'pos',[top_plot_ylabel_x, current_ylabel_pos(2), current_ylabel_pos(3)]);

line(tap_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(thermal_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(squeeze_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(slow_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(fast_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);


%% Third Subplot - PDC
ax(3) = subplot(7,1,3);
hold on;
plot(relevant_x, object.data.biotac(RIGHT).pdc(start_ind:end_ind), 'Color', [106 14 152]/255, 'LineWidth',plot_line_width);
plot(relevant_x, object.data.biotac(LEFT).pdc(start_ind:end_ind), 'Color', [205 120 247]/255, 'LineWidth',plot_line_width);
%Ylabel Alignment
y = ylabel('$P_{DC}$','Interpreter','latex','FontSize',ylabel_font_size, 'VerticalAlignment','middle');
set(get(gca,'YLabel'),'Rotation',0.0);
current_ylabel_pos = get(y,'pos'); % Read position [x y z] 
set(y,'pos',[top_plot_ylabel_x, current_ylabel_pos(LEFT), current_ylabel_pos(3)]) 
set(gca,'XTickLabel', '');
%Dotted Lines
line(tap_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(thermal_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(squeeze_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(slow_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(fast_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);


%% Fourth Subplot - PAC
ax(4) = subplot(7,1,4);
hold on;
sample_pac_time_sec = sample_time_sec/22;
relevant_pac_x = [center_grip_resize_ind:sample_pac_time_sec:done_resize_ind];
plot(relevant_pac_x, object.data.biotac(RIGHT).pac_flat(start_ind*22:end_ind*22), 'Color', [188 5 11]/255, 'LineWidth',plot_line_width);
plot(relevant_pac_x, object.data.biotac(LEFT).pac_flat(start_ind*22:end_ind*22), 'Color', [242 101 105]/255, 'LineWidth',plot_line_width);
%Ylabel Alignment
y = ylabel('$P_{AC}$','Interpreter','latex','FontSize',ylabel_font_size, 'VerticalAlignment','middle');
set(get(gca,'YLabel'),'Rotation',0.0); 
current_ylabel_pos = get(y,'pos'); % Read position [x y z] 
set(y,'pos',[top_plot_ylabel_x, current_ylabel_pos(2), current_ylabel_pos(3)]) 
set(gca,'XTickLabel', '');
%Dotted Lines
line(tap_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(thermal_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(squeeze_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(slow_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(fast_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);





 
%% Fifth Subplot - TDC
% ax(5) = subplot(7,1,5);
% hold on;
% plot(relevant_x, object.data.biotac(RIGHT).raw_tdc(start_ind:end_ind), 'Color', [10 156 63]/255, 'LineWidth',plot_line_width);
% plot(relevant_x, object.data.biotac(LEFT).raw_tdc(start_ind:end_ind), 'Color', [52 218 113]/255, 'LineWidth',plot_line_width);
% %Ylabel Alignment
% scale_factor = 0.3;
% 
% %yaxis = ylim();
% %ymax = yaxis(2)*(1+scale_factor);
% %ymin = yaxis(1)*(1-scale_factor);
% 
% ylim([1500 2500]) 
% y = ylabel('$T_{DC}$','Interpreter','latex','FontSize',ylabel_font_size, 'VerticalAlignment','middle');
% set(get(gca,'YLabel'),'Rotation',0.0);
% current_ylabel_pos = get(y,'pos'); % Read position [x y z] 
% set(y,'pos',[top_plot_ylabel_x, current_ylabel_pos(2), current_ylabel_pos(3)]); 
% set(gca,'XTickLabel', '');
% 
% 
% %Dotted Lines
% line(tap_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
% line(thermal_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
% line(squeeze_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
% line(slow_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
% line(fast_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);






%% Sixth Subplot - TAC
ax(5) = subplot(7,1,5);
hold on;
plot(relevant_x, object.data.biotac(RIGHT).tac(start_ind:end_ind), 'Color', [228 118 8]/255, 'LineWidth',plot_line_width);
plot(relevant_x, object.data.biotac(LEFT).tac(start_ind:end_ind), 'Color', [253 189 125]/255, 'LineWidth',plot_line_width);
%Ylabel Alignment
y = ylabel('$T_{AC}$','Interpreter','latex','FontSize',ylabel_font_size, 'VerticalAlignment','middle');
set(get(gca,'YLabel'),'Rotation',0.0);
current_ylabel_pos = get(y,'pos'); % Read position [x y z] 
set(y,'pos',[top_plot_ylabel_x, current_ylabel_pos(2), current_ylabel_pos(3)]) 
set(gca,'XTickLabel', '');
%Dotted Lines
line(tap_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(thermal_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(squeeze_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(slow_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(fast_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);





%% Seventh Subplot - Electrode Left
ax(6) = subplot(7,1,6);
hold on;
plot(relevant_x,object.data.biotac(LEFT).electrodes(start_ind:end_ind, :), 'LineWidth',plot_line_width);
%Ylabel Alignment
y = ylabel('$E_{1:19, L}$','Interpreter','latex','FontSize',ylabel_font_size, 'VerticalAlignment','middle');
set(get(gca,'YLabel'),'Rotation',0.0);
current_ylabel_pos = get(y,'pos'); % Read position [x y z] 
set(y,'pos',[top_plot_ylabel_x, current_ylabel_pos(2), current_ylabel_pos(3)]) 
set(gca,'XTickLabel', '');
%Dotted Lines
line(tap_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(thermal_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(squeeze_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(slow_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(fast_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);





%% Eighth Subplot - Electrode Right
ax(7) = subplot(7,1,7);
hold on;
plot(relevant_x,object.data.biotac(RIGHT).electrodes(start_ind:end_ind, :), 'LineWidth',plot_line_width);
y = ylabel('$E_{1:19, R}$','Interpreter','latex','FontSize',ylabel_font_size, 'VerticalAlignment','middle');
set(get(gca,'YLabel'),'Rotation',0.0); 
current_ylabel_pos = get(y,'pos'); % Read position [x y z] 
set(y,'pos',[top_plot_ylabel_x, current_ylabel_pos(2), current_ylabel_pos(3)])

line(tap_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(thermal_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(squeeze_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(slow_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);
line(fast_slide_list, ylim,'LineStyle','--', 'Color',[.6 .6 .6]);




xlabel('Time (s)', 'FontSize', 14);
linkaxes([ax(8) ax(7) ax(6) ax(5)  ax(4) ax(3) ax(2) ax(1)],'x');
xlim([center_grip_resize_ind done_resize_ind]);
%end
