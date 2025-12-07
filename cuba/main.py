from manim import *
import numpy as np
import scipy.stats as stats
import math

# Nice color ramp, cool (purplish) to hot (yellowish) RGB color values
COLOR_RAMP = [
    rgb_to_color([57 / 255, 0.0, 153 / 255]),
    rgb_to_color([158 / 255, 0.0, 89 / 255]),
    rgb_to_color([1.0, 0.0, 84 / 255]),
    rgb_to_color([1.0, 84 / 255, 0.0]),
    rgb_to_color([1.0, 189 / 255, 0.0])
]


def PDF_bivariate_normal(x_1, x_2, mu_1=0., mu_2=0., sigma_1=1., sigma_2=1., rho=0.):
    '''
    General form of probability density function of bivariate normal distribution
    '''
    normalizing_const = 1 / (2 * math.pi * sigma_1 * sigma_2 * math.sqrt(1 - rho ** 2))
    exp_coeff = -(1 / (2 * (1 - rho ** 2)))
    A = ((x_1 - mu_1) / sigma_1) ** 2
    B = -2 * rho * ((x_1 - mu_1) / sigma_1) * ((x_2 - mu_2) / sigma_2)
    C = ((x_2 - mu_2) / sigma_2) ** 2

    return normalizing_const * math.exp(exp_coeff * (A + B + C))


def sampleNormalDistribution(mu, sigma, n):
    rv = stats.multivariate_normal(mu, sigma, allow_singular=True)

    # 3. Sample n points
    samples = rv.rvs(size=n)
    return samples


class StandardBivariateNormal(ThreeDScene):
    '''
    Plots the surface of the probability density function of the standard
    bivariate normal distribution
    '''

    def construct(self):
        ax = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[0, 5, 0.1]
        )
        x_label = ax.get_x_axis_label(r'x')
        y_label = ax.get_y_axis_label(r'y', edge=UP, buff=0.2)
        z_label = ax.get_z_axis_label(r'\phi(x, y)', buff=0.2)
        axis_labels = VGroup(x_label, y_label, z_label)

        mean = [0, 0, 0]
        cov = [
            [0.5, 0, 0],  # Correlations with x
            [0, 0.5, 0],  # Correlations with y
            [0, 0, 0.002]  # Correlations with z
        ]

        self.GROUND_TRUTH = [1, 1, 0.02]

        mu_x = ValueTracker(mean[0])
        mu_y = ValueTracker(mean[1])
        sigma_1 = ValueTracker(cov[0][0])
        sigma_2 = ValueTracker(cov[1][1])
        rho = ValueTracker(1)
        resolution_fa = 50

        self.add(ax, axis_labels)
        self.set_camera_orientation(
            phi=75 * DEGREES,
            theta=-70 * DEGREES,
            frame_center=[2, 0, 2],
            zoom=0.75)

        ## sampling axes
        SAMPLE_AXIS_POSITION = (8, 0, 5)
        ax_sample = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-0.2, 0.2, 0.05]
        ).move_to(SAMPLE_AXIS_POSITION).scale(0.7).rotate(-0.3, about_point=SAMPLE_AXIS_POSITION, axis=Z_AXIS).rotate(
            -0.07, about_point=SAMPLE_AXIS_POSITION, axis=Y_AXIS)

        x_label_sample = ax.get_x_axis_label(r'x')
        y_label_sample = ax.get_y_axis_label(r'y', buff=0.2)
        z_label_sample = ax.get_z_axis_label(r'\alpha', buff=0.2)

        axis_labels = VGroup(x_label, y_label, z_label)
        self.add(ax_sample, axis_labels)

        distribution = always_redraw(
            lambda: Surface(
                lambda u, v: ax.c2p(
                    u, v, PDF_bivariate_normal(
                        u, v, mu_1=mu_x.get_value(), mu_2=mu_y.get_value(),  sigma_1=sigma_1.get_value(), sigma_2=sigma_2.get_value()
                    )
                ),
                resolution=(resolution_fa, resolution_fa),
                u_range=[-4.5, 4.5],
                v_range=[-4.5, 4.5],
                fill_opacity=0.7
            ).set_fill_by_value(
                axes=ax,
                # Utilize color ramp colors with, higher values are "warmer"
                colors=[(COLOR_RAMP[0], 0),
                        (COLOR_RAMP[1], 0.05),
                        (COLOR_RAMP[2], 0.1),
                        (COLOR_RAMP[3], 0.15),
                        (COLOR_RAMP[4], 0.2)]
            )
        )

        # 1. Create the tracked vector
        # Create ONE combined updater for the whole panel
        stats_panel = always_redraw(lambda:
                                    VGroup(
                                        # --- 1. The Mu Vector ---
                                        VGroup(
                                            MathTex(r"\boldsymbol{\mu} = "),
                                            MobjectMatrix(
                                                [[
                                                    DecimalNumber(mu_x.get_value(), num_decimal_places=2,
                                                                  include_sign=True, color=RED)
                                                ],
                                                    [
                                                        DecimalNumber(mu_y.get_value(), num_decimal_places=2,
                                                                      include_sign=True, color=BLUE)
                                                    ]],
                                                v_buff=0.6, h_buff=0.8, element_alignment_corner=ORIGIN
                                            )
                                        ).arrange(RIGHT),

                                        # --- 2. The Sigma Matrix ---
                                        VGroup(
                                            MathTex(r"\Sigma = "),
                                            MobjectMatrix(
                                                [[
                                                    # Top-Left: Var X
                                                    DecimalNumber(sigma_1.get_value(), num_decimal_places=2,
                                                                  include_sign=True, color=RED),
                                                    # Top-Right: Covariance
                                                    DecimalNumber(
                                                        rho.get_value() * np.sqrt(sigma_1.get_value()) * np.sqrt(
                                                            sigma_2.get_value()),
                                                        num_decimal_places=2, include_sign=True, color=PURPLE
                                                    )
                                                ],
                                                    [
                                                        # Bottom-Left: Covariance
                                                        DecimalNumber(
                                                            rho.get_value() * np.sqrt(sigma_1.get_value()) * np.sqrt(
                                                                sigma_2.get_value()),
                                                            num_decimal_places=2, include_sign=True, color=PURPLE
                                                        ),
                                                        # Bottom-Right: Var Y
                                                        DecimalNumber(sigma_2.get_value(), num_decimal_places=2,
                                                                      include_sign=True, color=BLUE)
                                                    ]],
                                                v_buff=0.6, h_buff=1.8, element_alignment_corner=ORIGIN,
                                                left_bracket="[", right_bracket="]"
                                            )
                                        ).arrange(RIGHT)

                                    )
                                    .arrange(DOWN, buff=0.5, aligned_edge=RIGHT)  # Stack them vertically
                                    .scale(0.8)  # Scale the whole group
                                    .to_corner(DR)  # <--- POSITIONING HAPPENS HERE
                                    )

        # Add it once
        self.add_fixed_in_frame_mobjects(stats_panel)
        self.play(Create(distribution))

        self.wait(2)

        # Now play your animation...
        for i in range(3):
            mean, cov = self.sample_and_recalculate(ax_sample, mean, cov)

            # Note: Calculate the target rho value for the animation
            target_rho = cov[0][1] / (math.sqrt(cov[0][0]) * math.sqrt(cov[1][1]))

            self.play(
                mu_x.animate.set_value(mean[0]),
                mu_y.animate.set_value(mean[1]),
                sigma_1.animate.set_value(max(math.sqrt(cov[0][0]), 0.1)),
                sigma_2.animate.set_value(max(math.sqrt(cov[1][1]), 0.1)),
                rho.animate.set_value(target_rho),
                run_time=2
            )
            self.wait(1)

        self.wait(5)

        '''

        # Set up animation
        self.add(ax, axis_labels)
        self.set_camera_orientation(
            phi=75*DEGREES,
            theta=-70*DEGREES,
            frame_center=[0, 0, 2],
            zoom=0.75)
        # Begin animation
        self.wait(1)
        # self.play(
        #     sigma_1.animate.set_value(0.5),
        #     sigma_2.animate.set_value(0.5),
        #     run_time=2,
        #     rate_func=rate_functions.smooth
        # )
        self.wait(2)
        '''

    def sample_and_recalculate(self, ax_sample, mu, sigma):
        # remove provious all_points and highlight_points if they exist
        if hasattr(self, 'all_points'):
            self.play(Uncreate(self.all_points, run_time=0.3))

        curr_sample = sampleNormalDistribution(mu, sigma, 100)
        self.all_points = VGroup()
        for i in curr_sample:
            print(i)
            point = Dot3D(
                ax_sample.c2p(
                    i[0],
                    i[1],
                    i[2]
                ),
                radius=0.03,
                color=COLOR_RAMP[4]
            )
            self.all_points.add(point)
        self.play(Create(self.all_points))
        self.wait(1)

        # highlight n points closest to ground truth
        distances = np.linalg.norm(curr_sample - np.array(self.GROUND_TRUTH), axis=1)
        closest_n = np.argsort(distances)[:20]

        new_mu = np.mean(curr_sample[closest_n], axis=0)
        new_sigma = np.cov(curr_sample[closest_n], rowvar=False)
        self.highlight_points = VGroup()
        for i in closest_n:
            point = Dot3D(
                ax_sample.c2p(
                    curr_sample[i][0],
                    curr_sample[i][1],
                    curr_sample[i][2]
                ),
                radius=0.05,
                color=RED
            )
            self.highlight_points.add(point)
        self.play(Indicate(self.highlight_points, scale_factor=1.5))
        self.wait(1)

        target_point = (8, 0, 1)

        # 2. Create a list of animations using list comprehension
        animations = [p.animate.move_to(target_point) for p in self.highlight_points]

        # 3. Unpack the list into self.play
        self.play(*animations, run_time=1)
        self.remove(self.highlight_points)

        return new_mu, new_sigma
