for _ in range(10):
        z_data_p.append(nn.generate_random(z_rand_type, z_rand_params, shape=[minibatch_size, z_size]))

d_is_sequence_probs = []

if k == 0:
                    subject_id = subjects[0]
                    skeleton_data, d_is_sequence_probs = \
                        generate_skeletons(input_data, input_past_data, z_data_p)
                    



 if k > 0:
                d_losses.append(d_training_loss / current_batch_size)
                g_losses.append(g_training_loss / current_batch_size)
                d_losses_prob.append(d_training_loss_prob / current_batch_size) 




if epoch > 20: # Ignore first couple of epochs.
                prob_count = -1. # Don't count ground truth.
                for z_prob in d_is_sequence_probs:
                    if z_prob >= 0.5:
                        prob_count += 1.
                current_prob = prob_count / (len(d_is_sequence_probs) - 1)
                
                if (current_prob >= g_best_prob) and (current_prob > 0.):
                    save_model_and_video = True
                    g_best_prob = current_prob
                    g_best_prob_epoch = epoch







source = SourceFactory(dataset, args.camera_calibration_file)
sensor = source.create_sensor()
body_inf = source.create_body()

skeleton2D = Skeleton2D(sensor, body_inf)

