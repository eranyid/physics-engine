extends Node
## Analytic ray–sphere: center at z=5, r=0.5 → hit distance 4.5 along +Z.

const EXPECTED_T := 4.5
const T_TOL := 1e-4

func _ready() -> void:
	if not ClassDB.class_exists("PePhysicsWorld"):
		push_error("ci_raycast_test: PePhysicsWorld missing")
		get_tree().quit(2)
		return
	var world: Node = ClassDB.instantiate("PePhysicsWorld")
	add_child(world)
	world.set("fixed_dt", 0.0)
	world.set("interpolate", false)

	var id: int = int(world.call("add_sphere_body", 1.0, 0.5, 0.0, 0.0, 5.0, 0.0, 0.0))
	if id < 0:
		push_error("ci_raycast_test: sphere create failed")
		get_tree().quit(2)
		return

	var hit: Dictionary = world.call(
		"raycast", Vector3(0, 0, 0), Vector3(0, 0, 1), 100.0
	)
	print("GODOT_CI_RAYCAST hit=", hit)
	if not hit.get("hit", false):
		push_error("ci_raycast_test: expected a hit")
		get_tree().quit(1)
		return
	var t: float = float(hit.get("distance", -1.0))
	if abs(t - EXPECTED_T) > T_TOL:
		push_error("ci_raycast_test: distance %s want %s ± %s" % [t, EXPECTED_T, T_TOL])
		get_tree().quit(1)
		return
	if int(hit.get("body", -1)) != id:
		push_error("ci_raycast_test: body index mismatch")
		get_tree().quit(1)
		return
	print("GODOT_CI_RAYCAST_OK t=", t)
	get_tree().quit(0)
