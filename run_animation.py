from vicsek_model.vicsek import Vicsek
from vicsek_model.animate import Animate

if __name__ == "__main__":
    vicsek = Vicsek(
        container_dimension=40,
        num_birds=20,
        velocity=0.5,
        noise=0,
        birds=None,
        search_radius=0.5,
    )
    animate = Animate(model=vicsek, figsize=(10, 10), interval_between_frames=100)
    animate.main(time_steps=1000)
