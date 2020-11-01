    def Network_initiation

        pass

        else:
        self.pred_dict, residue_loss_no_gt = architecture.get_direct_plane_loss_model(
            scope='DPLM',
            P=self.P,
            n_max_instances=n_max_instances,
            # gt_dict=self.gt_dict,
            is_training=self.is_training,
            bn_decay=self.bn_decay
        )
        self.total_loss = tf.reduce_mean(residue_loss_no_gt)
        self.total_miou_loss = tf.zeros(shape=[], dtype=tf.float32)
        self.total_normal_loss = tf.zeros(shape=[], dtype=tf.float32)
        self.total_type_loss = tf.zeros(shape=[], dtype=tf.float32)
        self.total_residue_loss = tf.zeros(shape=[], dtype=tf.float32)
        self.total_parameter_loss = tf.zeros(shape=[], dtype=tf.float32)


    def get_calculate_plane_loss_model(scope, P, n_max_instances, is_training, bn_decay):
        '''
            Inputs:
                - P: BxNx3 tensor, the input point cloud
                - K := n_max_instances
            Outputs: a dict, containing
                - W: BxNxK, segmentation instances, fractional
                - normal_per_point: BxNx3, normal per point
                - type_per_point: BxNxT, type per points. NOTE: this is before taking softmax!
                - parameters - a dict, each entry is a BxKx... tensor
        '''

        n_registered_primitives = 1 #plane only
        with tf.variable_scope(scope):
            net_results = build_pointnet2_seg('est_net', X=P, out_dims=[n_max_instances, 3, n_registered_primitives],
                                              is_training=is_training, bn_decay=bn_decay)
            W, normal_per_point, type_per_point = net_results
        W = tf.nn.softmax(W, axis=2)  # BxNxK
        normal_per_point = tf.nn.l2_normalize(normal_per_point, axis=2)  # BxNx3

        fitter_feed = {
            'P': P,
            'W': W,
            'normal_per_point': normal_per_point,
        }
        parameters = {}

        for fitter_cls in fitter_factory.get_all_fitter_classes():
            fitter_cls.compute_parameters(fitter_feed, parameters)

        residue_losses = []
        for fitter_cls in fitter_factory.get_all_fitter_classes():
            residue_per_point = fitter_cls.compute_residue_loss_no_gt(parameters,
                                                                         P)  # BxKxKxN'
            # residue_per_point = fitter_cls.compute_residue_loss_pairwise(parameters, gt_dict['points_per_instance']) # BxKxKxN'
            residue_avg = tf.reduce_mean(residue_per_point, axis=3)  # BxKxK
            # residue_avg[b, k1, k2] is roughly the distance between gt instance k1 and predicted instance k2
            residue_losses.append(residue_avg)
            residue_loss_no_gt = tf.reduce_sum(residue_losses)


        return {
            'W': W,
            'normal_per_point': normal_per_point,
            'type_per_point': type_per_point,
            'parameters': parameters,
        }, residue_loss_no_gt


    def compute_residue_loss_no_gt(parameters, P):
        return PlaneFitter.compute_residue_single(
            *adaptor_pairwise([parameters['plane_n'], parameters['plane_c']]),
            adaptor_P_pairwise(P)

    def adaptor_P_pairwise(P):
        # P_gt is BxKxN'x3, making it BxKxKxN'x3
        # return tf.tile(tf.expand_dims(P_gt, axis=2), [1, 1, tf.shape(P_gt)[1], 1, 1])
        return tf.expand_dims(tf.expand_dims(P, axis=1), axis=2)

    def weighted_plane_fitting_with_loss(P, W):
        # P - BxNx3
        # W - BxN
        # Returns n, c, with n - Bx3, c - B
        WP = P * tf.expand_dims(W, axis=2) # BxNx3
        W_sum = tf.reduce_sum(W, axis=1) # B
        P_weighted_mean = tf.reduce_sum(WP, axis=1) / tf.maximum(tf.expand_dims(W_sum, 1), DIVISION_EPS) # Bx3
        A = P - tf.expand_dims(P_weighted_mean, axis=1) # BxNx3
        n = solve_weighted_tls(A, W) # Bx3
        c = tf.reduce_sum(n * P_weighted_mean, axis=1)
        loss = tf.square(tf.reduce_sum(P_weighted_mean * n, axis=-1) - c)
        return n, c, loss