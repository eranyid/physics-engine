extends Node
## Drop a box onto a ground plane; expect contact_started at least once.

const DT := 1.0 / 60.0
const STEPS := 180

var started_count: int = 0

func _on_contact_started(_a: int, _b: int) -> void:
	started_count += 1

func _ready() -> void:
	if not ClassDB.class_exists("PePhysicsWorld"):
		push_error("ci_contact_signal_test: PePhysicsWorld missing")
		get_tree().quit(2)
		return
	var world: Node = ClassDB.instantiate("PePhysicsWorld")
	add_child(world)
	world.set("fixed_dt", 0.0)
	world.set("interpolate", false)
	world.connect("contact_started", Callable(self, "_on_contact_started"))

	world.call("add_ground_plane", 0.0, 0.0, 1.0)
	var id: int = int(world.call("add_box_body", 1.0, 0.25, 0.25, 0.25, 0.0, 2.0, 0.0, 0.0, 0.5))
	if id < 0:
		push_error("ci_contact_signal_test: box create failed")
		get_tree().quit(2)
		return

	for _i in STEPS:
		world.call("step_once", DT)

	print("GODOT_CI_CONTACT started_count=", started_count)
	if started_count < 1:
		push_error("ci_contact_signal_test: contact_started never fired")
		get_tree().quit(1)
		return
	print("GODOT_CI_CONTACT_OK")
	get_tree().quit(0)
