from diffusers import DPMSolverMultistepScheduler

def create_scheduler():
    #ddpm 2M karras
    return DPMSolverMultistepScheduler(
        num_train_timesteps = 1000,
        beta_start = 0.0001,
        beta_end = 0.02,
        beta_schedule="linear",
        algorithm_type = "dpmsolver++",
        solver_order=2, 
        use_karras_sigmas = True
    )