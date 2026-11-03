extends Node3D
## Interactive demo: box stack, pendulum, and a raycast probe each frame.

@onready var world: Node = $PePhysicsWorld
@onready var ray_label: Label3D = $RayLabel

func _ready() -> void:
	if world == null:
		return
	world.set("fixed_dt", 1.0 / 60.0)
	world.set("interpolate", true)
	world.call("add_ground_plane", 0.0, 0.0, 0.8)

	# Stack of three boxes.
	for i in 3:
		var y := 0.3 + float(i) * 0.55
		world.call("add_box_body", 1.0, 0.25, 0.25, 0.25, 0.0, y, 0.0, 0.0, 0.5)

	# Pendulum bob + distance joint to world.
	var bob: int = int(world.call("add_sphere_body", 1.0, 0.15, 2.0, 1.0, 0.0, 0.0, 0.2))
	world.call("add_distance_joint", bob, -1, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 90.0, 0.7)

func _process(_delta: float) -> void:
	if world == null or ray_label == null:
		return
	var hit: Dictionary = world.call(
		"raycast", Vector3(0, 3, 0), Vector3(0, -1, 0), 10.0
	)
	if hit.get("hit", false):
		ray_label.text = "ray hit t=%.3f body=%s" % [hit.get("distance", 0.0), hit.get("body", -1)]
	else:
		ray_label.text = "ray miss"
