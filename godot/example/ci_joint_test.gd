extends Node
## Distance-joint pendulum: bob must stay near rest length (not free-fall).

const DT := 1.0 / 60.0
const STEPS := 180
const L := 1.0

func _ready() -> void:
	if not ClassDB.class_exists("PePhysicsWorld"):
		push_error("ci_joint_test: PePhysicsWorld missing")
		get_tree().quit(2)
		return
	var world: Node = ClassDB.instantiate("PePhysicsWorld")
	add_child(world)
	world.set("fixed_dt", 0.0)
	world.set("interpolate", false)

	# Bob at (L, 0, 0); distance joint of length L to world origin.
	var bob_id: int = int(world.call("add_sphere_body", 1.0, 0.1, L, 0.0, 0.0, 0.0, 0.0))
	if bob_id < 0:
		push_error("ci_joint_test: bob create failed")
		get_tree().quit(2)
		return
	var jid: int = int(world.call(
		"add_distance_joint", bob_id, -1,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		L, 120.0, 0.7
	))
	if jid < 0:
		push_error("ci_joint_test: joint create failed")
		get_tree().quit(2)
		return

	for _i in STEPS:
		world.call("step_once", DT)

	var p: Vector3 = world.call("get_body_position", bob_id)
	var dist: float = p.length()
	print("GODOT_CI_JOINT pos=", p, " dist=", dist)
	# Soft joint: allow 15% stretch; free-fall after 3 s would be tens of metres.
	if dist > L * 1.15 or dist < L * 0.5:
		push_error("ci_joint_test: pendulum length out of band: %s (L=%s)" % [dist, L])
		get_tree().quit(1)
		return
	if p.y < -5.0:
		push_error("ci_joint_test: bob free-fell (y=%s)" % p.y)
		get_tree().quit(1)
		return
	print("GODOT_CI_JOINT_OK")
	get_tree().quit(0)
